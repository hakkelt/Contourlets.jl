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
_base_array_type(::Type{T}) where {T} = T
_base_array_type(::Type{SubArray{T, N, P, I, L}}) where {T, N, P, I, L} = _base_array_type(P)

function _scratch_like(ref::R, m::Int, n::Int) where {R <: AbstractMatrix}
    a = _active_arena()
    B = _base_array_type(R)
    if a isa ScratchArena{B}
        return _arena_take!(a, ref, m, n)
    end
    return similar(ref, m, n)
end

function _allocate_zeros(::Type{M}, Td::Type, dims::Tuple) where {M <: AbstractMatrix}
    T_concrete = M isa UnionAll ? M{Td} : M
    if T_concrete <: Array
        return zeros(Td, dims...)
    end
    dummy = similar(T_concrete, (1, 1))
    arr = similar(dummy, Td, dims)
    return fill!(arr, zero(Td))
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
struct ContourletWorkspace{Td <: Number, Tf <: AbstractFloat, M <: AbstractMatrix{Td}, Q, L, DF}
    params::ContourletParams{Tf}     # real filter precision Tf
    image_size::Tuple{Int, Int}

    # Grow-once pool for all temporary buffers (LP coarse/bandpass, transients).
    # Separate arenas for forward and inverse passes because they traverse the
    # pyramid levels in opposite orders, yielding different buffer size sequences.
    fwd_scratch::ScratchArena{M}
    inv_scratch::ScratchArena{M}

    # Pre-built (possibly device-resident) DFB ladder filter for CT workspaces.
    # Avoids re-uploading the same filter vector on every ct_forward!/ct_inverse! call.
    # `nothing` for non-ladder qfp or NSCT workspaces.
    dfb_f_cache::DF

    # Per-level upsampled NSDFB filters (empty for a CT workspace, which uses the
    # un-upsampled ladder filter directly).  Precomputed once at construction.
    qup_cache::Q

    # Per-level à trous upsampled LP analysis/synthesis filters for the NSP stage
    # (the scaled synthesis `g` serves both decompose and reconstruct).  Empty for
    # a CT workspace (the decimated LP uses the base filters directly).
    lp_h_cache::L
    lp_g_cache::L

    # FFT Caches for D2 equivalent-filter NSDFB (empty for CT workspaces).
    nsdfb_H_cache::Vector{Vector{Matrix{Complex{Tf}}}}
    nsdfb_G_cache::Vector{Vector{Matrix{Complex{Tf}}}}
    fft_plan_fwd::Any
    fft_plan_inv::Any
    fft_buffer_c::Matrix{Complex{Tf}}
    fft_spectrum_bp::Matrix{Complex{Tf}}

    # FFTW thread state frozen at construction time.  The FFT plans are bound to
    # this thread count and cannot be changed per-call; a mismatch between this
    # field and a per-call threading kwarg triggers a one-time @warn.
    fft_threaded::Bool
end

function ContourletWorkspace{Td, Tf, M}(
        params::ContourletParams{Tf},
        image_size::Tuple{Int, Int},
        fwd_scratch::ScratchArena{M},
        inv_scratch::ScratchArena{M},
        qup_cache::Q,
        lp_h_cache::L,
        lp_g_cache::L,
        nsdfb_H_cache::Vector{Vector{Matrix{Complex{Tf}}}},
        nsdfb_G_cache::Vector{Vector{Matrix{Complex{Tf}}}},
        fft_plan_fwd::Any,
        fft_plan_inv::Any,
        fft_buffer_c::Matrix{Complex{Tf}},
        fft_spectrum_bp::Matrix{Complex{Tf}},
        fft_threaded::Bool = false,
        dfb_f_cache::DF = nothing
    ) where {Td, Tf, M, Q, L, DF}
    return ContourletWorkspace{Td, Tf, M, Q, L, DF}(
        params, image_size, fwd_scratch, inv_scratch, dfb_f_cache,
        qup_cache, lp_h_cache, lp_g_cache,
        nsdfb_H_cache, nsdfb_G_cache, fft_plan_fwd, fft_plan_inv, fft_buffer_c, fft_spectrum_bp,
        fft_threaded
    )
end

_device_qup_cache(::Type{M}, qup_cache) where {M <: AbstractMatrix} = qup_cache
_device_lp_cache(::Type{M}, lp_cache) where {M <: AbstractMatrix} = lp_cache
_device_dfb_filter(::Type{M}, f) where {M <: AbstractMatrix} = f

Base.eltype(::ContourletWorkspace{Td}) where {Td} = Td

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

julia> ws isa ContourletWorkspace
true
```
"""
make_workspace(
    T::Type{<:Number}, image_size::Tuple{Int, Int},
    params::ContourletParams; kwargs...
) =
    make_workspace(params, image_size; T = T, kwargs...)

"""
    make_workspace(image::AbstractMatrix, params::ContourletParams) -> ContourletWorkspace

Allocate a `ContourletWorkspace` using the array type and element type of `image`.
This is required for allocation-free execution on GPU arrays.
"""
make_workspace(
    image::AbstractMatrix, params::ContourletParams{Tp}; kwargs...
) where {Tp} =
    make_workspace(params, size(image); T = eltype(image), M = typeof(similar(image, promote_type(Tp, eltype(image)), 0, 0)), kwargs...)

function make_workspace(
        params::ContourletParams{Tp},
        image_size::Tuple{Int, Int};
        T::Type{<:Number} = Float64,
        M::Type{<:AbstractMatrix} = Matrix{promote_type(Tp, T)},
        prewarm::Bool = true
    ) where {Tp}
    Td = promote_type(Tp, T)         # data buffer type (may be complex)
    Tf = _filter_eltype(Td)          # real filter precision
    p2 = _convert_params(Tf, params)
    # Build the ladder filter once and upload to device (no-op on CPU).
    dfb_f = is_ladder(p2.dfb_filters) ? _ladder_modulate(Tf.(p2.dfb_filters.f_ladder)) : nothing
    dfb_f_cache = _device_dfb_filter(M, dfb_f)
    # CT uses the un-upsampled ladder/LP filters directly, so no per-level filter
    # caches are needed.
    ws = ContourletWorkspace{Td, Tf, M}(
        p2, image_size, ScratchArena{M}(), ScratchArena{M}(),
        _NSDFBFilters[], Vector{Tf}[], Vector{Tf}[],
        Vector{Matrix{Complex{Tf}}}[], Vector{Matrix{Complex{Tf}}}[], nothing, nothing, Matrix{Complex{Tf}}(undef, 0, 0), Matrix{Complex{Tf}}(undef, 0, 0),
        false, dfb_f_cache
    )
    # Grow the scratch arena to its steady-state size now (forward + inverse), so
    # the first user call is already allocation-free.
    prewarm && _prewarm_ct!(ws)
    return ws
end

"""
    make_nsct_workspace(params::ContourletParams, image_size::Tuple{Int,Int}; T=Float64,
                        threading=Auto()) -> ContourletWorkspace
    make_nsct_workspace(T::Type, image_size::Tuple{Int,Int}, params::ContourletParams)
                        -> ContourletWorkspace

Allocate a workspace for the NSCT transforms.  All buffers have the full `image_size`
since the NSCT performs no downsampling.
The second form is a convenience overload with the element type as the first positional argument.

!!! note "FFT threading"
    The `threading` kwarg controls how many FFTW threads the pre-built FFT plans use.
    This choice is frozen at construction time: the plans cannot switch thread counts
    per-call.  Passing a conflicting `threading` kwarg to `nsct_forward!`/`nsct_inverse!`
    later will emit a one-time warning.  To change FFT threading, create a new workspace.
"""
make_nsct_workspace(
    T::Type{<:Number}, image_size::Tuple{Int, Int},
    params::ContourletParams; kwargs...
) =
    make_nsct_workspace(params, image_size; T = T, kwargs...)

"""
    make_nsct_workspace(image::AbstractMatrix, params::ContourletParams) -> ContourletWorkspace

Allocate a workspace for the NSCT transforms using the array type and element type of `image`.
This is required for allocation-free execution on GPU arrays.
"""
make_nsct_workspace(
    image::AbstractMatrix, params::ContourletParams{Tp}; kwargs...
) where {Tp} =
    make_nsct_workspace(params, size(image); T = eltype(image), M = typeof(similar(image, promote_type(Tp, eltype(image)), 0, 0)), kwargs...)

function make_nsct_workspace(
        params::ContourletParams{Tp},
        image_size::Tuple{Int, Int};
        T::Type{<:Number} = Float64,
        M::Type{<:AbstractMatrix} = Matrix{promote_type(Tp, T)},
        prewarm::Bool = true,
        threading::ThreadingPolicy = Auto()
    ) where {Tp}
    Td = promote_type(Tp, T)         # data buffer type (may be complex)
    Tf = _filter_eltype(Td)          # real filter precision
    p2 = _convert_params(Tf, params)
    J = p2.J
    # Precompute the 2-D NSDFB fan/parallelogram filter bundle (scale independent,
    # so identical for every LP level) and the à trous upsampled LP filters; the
    # `!` paths then never rebuild them per call.
    nsdfb_bundle = _nsdfb_filters(p2.dfb_filters, Tf)
    qup_cache = [nsdfb_bundle for _ in 1:J]
    lp = p2.lp_filters
    lp_h_cache = Vector{Tf}[upsample_filter(lp.h, 2^(j - 1)) for j in 1:J]
    lp_g_cache = Vector{Tf}[Tf(_NSP_SYNTH_SCALE) .* upsample_filter(lp.g, 2^(j - 1)) for j in 1:J]

    is_real = Td <: Real
    n1, n2 = image_size
    init_shape = is_real ? (n1 ÷ 2 + 1, n2) : (n1, n2)
    fft_buffer_c = zeros(Complex{Tf}, init_shape)
    fft_spectrum_bp = zeros(Complex{Tf}, init_shape)

    fft_threaded = false
    if M <: Array
        tmp_d = zeros(Td, n1, n2)
        fft_threaded = _use_threading(threading, Td)
        FFTW.set_num_threads(fft_threaded ? Threads.nthreads() : 1)
        fft_plan_fwd = is_real ? plan_rfft(tmp_d, flags = FFTW.MEASURE) : plan_fft(tmp_d, flags = FFTW.MEASURE)
        fft_plan_inv = is_real ? plan_irfft(fft_buffer_c, n1, flags = FFTW.MEASURE) : plan_ifft(fft_buffer_c, flags = FFTW.MEASURE)
    else
        fft_plan_fwd = nothing
        fft_plan_inv = nothing
    end

    nsdfb_H_cache = Vector{Vector{Matrix{Complex{Tf}}}}(undef, J)
    nsdfb_G_cache = Vector{Vector{Matrix{Complex{Tf}}}}(undef, J)
    for j in 1:J
        nsdfb_H_cache[j], nsdfb_G_cache[j] = _build_equivalent_filters(Tf, n1, n2, qup_cache[j], p2.L_array[j], is_real)
    end

    qup_cache = _device_qup_cache(M, qup_cache)
    lp_h_cache = _device_lp_cache(M, lp_h_cache)
    lp_g_cache = _device_lp_cache(M, lp_g_cache)

    ws = ContourletWorkspace{Td, Tf, M}(
        p2, image_size, ScratchArena{M}(), ScratchArena{M}(),
        qup_cache, lp_h_cache, lp_g_cache,
        nsdfb_H_cache, nsdfb_G_cache, fft_plan_fwd, fft_plan_inv, fft_buffer_c, fft_spectrum_bp,
        fft_threaded
    )
    prewarm && _prewarm_nsct!(ws; threading = threading)
    return ws
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
        total += 4 * J * n1 * n2          # coarse_j + bp_j + tmp_j + tmp2_j, full size each level
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
