# ContourletWorkspace — preallocated buffers for zero-per-iteration allocations.
#
# Design goals:
#   1. All temporary matrices are allocated once at construction time.
#   2. The mutating entry points ct_forward! / ct_inverse! / nsct_forward! /
#      nsct_inverse! accept a ContourletWorkspace and reuse its buffers.
#   3. Thread-safety: one workspace per thread; NOT safe for concurrent reuse.
#   4. Backward compatibility: non-! functions always allocate; workspace is opt-in.

"""
    ContourletWorkspace{T}

Preallocated temporary buffers for the Contourlet and NSCT transforms.  Create
with `make_workspace`; reuse across iterations with the mutating `!` entry points.

!!! warning "Thread safety"
    A `ContourletWorkspace` is **not thread-safe**.  Create a separate workspace
    per thread (e.g., `[make_workspace(params, sz) for _ in 1:Threads.nthreads()]`).
"""
struct ContourletWorkspace{T <: AbstractFloat}
    params::ContourletParams{T}
    image_size::Tuple{Int, Int}

    # Per-level intermediate buffers (length = J)
    coarse_bufs::Vector{Matrix{T}}   # coarse at each LP level
    bp_bufs::Vector{Matrix{T}}       # bandpass at each LP level

    # Scratch buffers for separable conv and reconstruction
    tmp_buf::Matrix{T}               # size = image_size (for in-place conv)
    current::Matrix{T}               # running coarse (size changes per level for CT)
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
julia> p = ContourletParams(J=2, L_array=[1,2])
julia> ws = make_workspace(p, (64, 64))
julia> typeof(ws)
ContourletWorkspace{Float64}
```
"""
make_workspace(
    T::Type{<:AbstractFloat}, image_size::Tuple{Int, Int},
    params::ContourletParams
) =
    make_workspace(params, image_size; T = T)

function make_workspace(
        params::ContourletParams{Tp},
        image_size::Tuple{Int, Int};
        T::Type{<:AbstractFloat} = Float64
    ) where {Tp}
    J = params.J
    n1, n2 = image_size
    coarse_bufs = Vector{Matrix{T}}(undef, J)
    bp_bufs = Vector{Matrix{T}}(undef, J)
    c1, c2 = n1, n2
    for j in 1:J
        d1, d2 = cld(c1, 2), cld(c2, 2)
        coarse_bufs[j] = zeros(T, d1, d2)
        bp_bufs[j] = zeros(T, c1, c2)
        c1, c2 = d1, d2
    end
    tmp_buf = zeros(T, n1, n2)
    current = zeros(T, n1, n2)
    Tp2 = promote_type(Tp, T)
    p2 = ContourletParams{Tp2}(
        params.J, params.L_array,
        FilterPair{Tp2}(
            Tp2.(params.lp_filters.h),
            Tp2.(params.lp_filters.g)
        ),
        QuincunxFilterPair{Tp2}(
            Tp2.(params.dfb_filters.h_q),
            Tp2.(params.dfb_filters.g_q),
            params.dfb_filters.c_h,
            params.dfb_filters.c_g
        )
    )
    return ContourletWorkspace{T}(p2, image_size, coarse_bufs, bp_bufs, tmp_buf, current)
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
    T::Type{<:AbstractFloat}, image_size::Tuple{Int, Int},
    params::ContourletParams
) =
    make_nsct_workspace(params, image_size; T = T)

function make_nsct_workspace(
        params::ContourletParams{Tp},
        image_size::Tuple{Int, Int};
        T::Type{<:AbstractFloat} = Float64
    ) where {Tp}
    J = params.J
    n1, n2 = image_size
    coarse_bufs = [zeros(T, n1, n2) for _ in 1:J]
    bp_bufs = [zeros(T, n1, n2) for _ in 1:J]
    tmp_buf = zeros(T, n1, n2)
    current = zeros(T, n1, n2)
    Tp2 = promote_type(Tp, T)
    p2 = ContourletParams{Tp2}(
        params.J, params.L_array,
        FilterPair{Tp2}(
            Tp2.(params.lp_filters.h),
            Tp2.(params.lp_filters.g)
        ),
        QuincunxFilterPair{Tp2}(
            Tp2.(params.dfb_filters.h_q),
            Tp2.(params.dfb_filters.g_q),
            params.dfb_filters.c_h,
            params.dfb_filters.c_g
        )
    )
    return ContourletWorkspace{T}(p2, image_size, coarse_bufs, bp_bufs, tmp_buf, current)
end

"""
    estimate_workspace_size(params::ContourletParams, image_size) -> Int

Return the total number of floating-point elements across all workspace buffers
(useful for memory planning before allocation).
"""
function estimate_workspace_size(
        params::ContourletParams,
        image_size::Tuple{Int, Int}
    )::Int
    J = params.J
    n1, n2 = image_size
    total = 0
    c1, c2 = n1, n2
    for j in 1:J
        d1, d2 = cld(c1, 2), cld(c2, 2)
        total += d1 * d2    # coarse buf
        total += c1 * c2    # bp buf
        c1, c2 = d1, d2
    end
    total += 2 * n1 * n2    # tmp_buf + current
    return total
end

"""
    workspace_clear!(ws::ContourletWorkspace)

Zero all numeric buffers in `ws`.  Call this when the algorithmic state should be
reset (e.g., when switching to a different image in the same iterative loop).
"""
function workspace_clear!(ws::ContourletWorkspace{T}) where {T}
    fill!(ws.tmp_buf, zero(T))
    fill!(ws.current, zero(T))
    for b in ws.coarse_bufs
        fill!(b, zero(T))
    end
    for b in ws.bp_bufs
        fill!(b, zero(T))
    end
    return ws
end
