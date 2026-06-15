# ContourletWorkspace — preallocated buffers for zero-per-iteration allocations.
#
# Design goals:
#   1. All temporary matrices are allocated once at construction time.
#   2. The mutating entry points ct_forward! / ct_inverse! / nsct_forward! /
#      nsct_inverse! accept a ContourletWorkspace and reuse its buffers.
#   3. Thread-safety: one workspace per thread; NOT safe for concurrent reuse.
#   4. Backward compatibility: non-! functions always allocate; workspace is opt-in.

# ── Scratch arena ─────────────────────────────────────────────────────────────
# A grow-once pool of `Matrix{T}` scratch buffers for the directional stage,
# whose transient buffers vary in number/size per node.  The allocation sequence
# is deterministic for fixed params + image size, so `reset!` at the start of a
# transform and `acquire!` hand back the same buffers on every subsequent call —
# zero allocations after the first.  Buffers are handed out by a monotonic cursor
# (never released mid-transform), so all live intermediates are distinct.

# Parametrised on the stored matrix type `M` so the same pool serves host
# (`Matrix`) and device (`CuMatrix`, …) buffers; new buffers are made with
# `similar(ref, …)` from a caller-supplied reference array.
mutable struct ScratchArena{M <: AbstractMatrix}
    bufs::Vector{M}
    cursor::Int
end
ScratchArena{M}() where {M <: AbstractMatrix} = ScratchArena{M}(M[], 0)

_arena_reset!(a::ScratchArena) = (a.cursor = 0; return a)

# Hand out a scratch `m×n` matrix shaped like `ref`, growing/resizing as needed.
function _arena_take!(a::ScratchArena{M}, ref, m::Int, n::Int) where {M}
    a.cursor += 1
    if a.cursor > length(a.bufs)
        push!(a.bufs, similar(ref, m, n)::M)
    elseif size(a.bufs[a.cursor]) != (m, n)
        a.bufs[a.cursor] = similar(ref, m, n)::M
    end
    return a.bufs[a.cursor]
end

# Task-local "active" arena.  The directional tree is a deep recursion of shared
# functions, so rather than thread an arena argument through all of them the host
# primitives (`_resamp`, `_extend2`, `_sefilter2`, the lifting broadcasts) draw
# transient buffers from whatever arena is active on the current task, via
# `_scratch_like`.  When no matching arena is active `_scratch_like` falls back to
# `similar` (so the non-workspace and GPU paths are unchanged).  The arena is
# typed on its matrix type, so a host arena only serves host arrays.
# Task-local ⇒ thread-safe.

_active_arena() = get(task_local_storage(), :_contourlets_scratch, nothing)

function _with_arena(f, arena)
    store = task_local_storage()
    prev = get(store, :_contourlets_scratch, nothing)
    store[:_contourlets_scratch] = arena
    try
        return f()
    finally
        store[:_contourlets_scratch] = prev
    end
end

# A scratch `m×n` matrix shaped like `ref`: drawn from the active arena when its
# stored matrix type matches `typeof(ref)`, otherwise `similar(ref, m, n)`.  The
# type match means a host arena only ever serves host arrays and a device arena
# only device arrays, so mixed/absent arenas fall back to `similar` safely.
function _scratch_like(ref::R, m::Int, n::Int) where {R <: AbstractMatrix}
    a = _active_arena()
    if a isa ScratchArena{R}
        return _arena_take!(a, ref, m, n)
    end
    return similar(ref, m, n)
end

"""
    ContourletWorkspace{Td, Tf}

Preallocated temporary buffers (data type `Td`, real filter precision `Tf`) for the
Contourlet and NSCT transforms.  Create with `make_workspace`; reuse across
iterations with the mutating `!` entry points.

!!! warning "Thread safety"
    A `ContourletWorkspace` is **not thread-safe**.  Create a separate workspace
    per thread (e.g., `[make_workspace(params, sz) for _ in 1:Threads.nthreads()]`).
"""
struct ContourletWorkspace{Td <: Number, Tf <: AbstractFloat}
    params::ContourletParams{Tf}     # real filter precision Tf
    image_size::Tuple{Int, Int}

    # Per-level intermediate buffers (length = J), at the data type Td.
    coarse_bufs::Vector{Matrix{Td}}  # coarse at each LP level
    bp_bufs::Vector{Matrix{Td}}      # bandpass at each LP level

    # Scratch buffers for separable conv and reconstruction
    tmp_buf::Matrix{Td}              # size = image_size (for in-place conv)
    tmp_buf2::Matrix{Td}             # second scratch (separable-conv intermediate)
    current::Matrix{Td}              # running coarse (size changes per level for CT)

    # Grow-once pool for the directional-stage transients (DFB / NSDFB).
    scratch::ScratchArena{Matrix{Td}}
end

"""
    make_workspace(params::ContourletParams, image_size::Tuple{Int,Int}; T=Float64)
    make_workspace(T::Type, image_size::Tuple{Int,Int}, params::ContourletParams)
                   -> ContourletWorkspace

Allocate a `ContourletWorkspace` for images of `image_size` with the given `params`.
The second form is a convenience overload with the element type as the first positional argument.

# Examples
```jldoctest
julia> using Contourlets

julia> p = ContourletParams(J = 2, L_array = [1, 2]);

julia> ws = make_workspace(p, (64, 64));

julia> typeof(ws)
ContourletWorkspace{Float64, Float64}
```
"""
make_workspace(
    T::Type{<:Number}, image_size::Tuple{Int, Int},
    params::ContourletParams
) =
    make_workspace(params, image_size; T = T)

function make_workspace(
        params::ContourletParams{Tp},
        image_size::Tuple{Int, Int};
        T::Type{<:Number} = Float64
    ) where {Tp}
    Td = promote_type(Tp, T)         # data buffer type (may be complex)
    Tf = _filter_eltype(Td)          # real filter precision
    J = params.J
    n1, n2 = image_size
    coarse_bufs = Vector{Matrix{Td}}(undef, J)
    bp_bufs = Vector{Matrix{Td}}(undef, J)
    c1, c2 = n1, n2
    for j in 1:J
        d1, d2 = cld(c1, 2), cld(c2, 2)
        coarse_bufs[j] = zeros(Td, d1, d2)
        bp_bufs[j] = zeros(Td, c1, c2)
        c1, c2 = d1, d2
    end
    tmp_buf = zeros(Td, n1, n2)
    tmp_buf2 = zeros(Td, n1, n2)
    current = zeros(Td, n1, n2)
    p2 = _convert_params(Tf, params)
    return ContourletWorkspace{Td, Tf}(
        p2, image_size, coarse_bufs, bp_bufs, tmp_buf, tmp_buf2, current, ScratchArena{Matrix{Td}}()
    )
end

"""
    make_nsct_workspace(params::ContourletParams, image_size::Tuple{Int,Int}; T=Float64)
    make_nsct_workspace(T::Type, image_size::Tuple{Int,Int}, params::ContourletParams)
                        -> ContourletWorkspace

Allocate a workspace for the NSCT transforms.  All buffers have the full `image_size`
since the NSCT performs no downsampling.
The second form is a convenience overload with the element type as the first positional argument.
"""
make_nsct_workspace(
    T::Type{<:Number}, image_size::Tuple{Int, Int},
    params::ContourletParams
) =
    make_nsct_workspace(params, image_size; T = T)

function make_nsct_workspace(
        params::ContourletParams{Tp},
        image_size::Tuple{Int, Int};
        T::Type{<:Number} = Float64
    ) where {Tp}
    Td = promote_type(Tp, T)         # data buffer type (may be complex)
    Tf = _filter_eltype(Td)          # real filter precision
    J = params.J
    n1, n2 = image_size
    coarse_bufs = [zeros(Td, n1, n2) for _ in 1:J]
    bp_bufs = [zeros(Td, n1, n2) for _ in 1:J]
    tmp_buf = zeros(Td, n1, n2)
    tmp_buf2 = zeros(Td, n1, n2)
    current = zeros(Td, n1, n2)
    p2 = _convert_params(Tf, params)
    return ContourletWorkspace{Td, Tf}(
        p2, image_size, coarse_bufs, bp_bufs, tmp_buf, tmp_buf2, current, ScratchArena{Matrix{Td}}()
    )
end

"""
    estimate_workspace_size(params::ContourletParams, image_size;
                            nonsubsampled=false) -> Int

Return the total number of floating-point elements across all workspace buffers
(useful for memory planning before allocation).  Pass `nonsubsampled=true` for
workspaces created with [`make_nsct_workspace`](@ref), whose per-level buffers
all have the full image size.
"""
function estimate_workspace_size(
        params::ContourletParams,
        image_size::Tuple{Int, Int};
        nonsubsampled::Bool = false
    )::Int
    J = params.J
    n1, n2 = image_size
    total = 0
    if nonsubsampled
        total += 2 * J * n1 * n2          # coarse + bp bufs, full size each level
    else
        c1, c2 = n1, n2
        for j in 1:J
            d1, d2 = cld(c1, 2), cld(c2, 2)
            total += d1 * d2    # coarse buf
            total += c1 * c2    # bp buf
            c1, c2 = d1, d2
        end
    end
    total += 3 * n1 * n2    # tmp_buf + tmp_buf2 + current
    return total
end

"""
    workspace_clear!(ws::ContourletWorkspace)

Zero all numeric buffers in `ws`.  Call this when the algorithmic state should be
reset (e.g., when switching to a different image in the same iterative loop).
"""
function workspace_clear!(ws::ContourletWorkspace{T}) where {T}
    fill!(ws.tmp_buf, zero(T))
    fill!(ws.tmp_buf2, zero(T))
    fill!(ws.current, zero(T))
    for b in ws.coarse_bufs
        fill!(b, zero(T))
    end
    for b in ws.bp_bufs
        fill!(b, zero(T))
    end
    return ws
end
