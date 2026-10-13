# CTWorkspace and NSCTWorkspace — preallocated buffers for zero-per-iteration
# allocations.
#
# Design goals:
#   1. All temporary matrices are allocated once at construction time.
#   2. The mutating entry points ct_forward! / ct_inverse! (with a CTWorkspace) and
#      nsct_forward! / nsct_inverse! (with an NSCTWorkspace) reuse their buffers.
#   3. Thread-safety: one workspace per thread; NOT safe for concurrent reuse.
#   4. Backward compatibility: non-! functions always allocate; workspace is opt-in.
#
# Scratch buffers come from the `ScratchArena` in `scratch_arena.jl`.

"""
    CTWorkspace{Td, Tf}

Preallocated buffers (data type `Td`, real filter precision `Tf`) for the
contourlet transform: the pyramid's scratch arenas and, per pyramid level with a
directional split, the DFB's equivalent filters, FFT buffers and plans.  Create
with [`make_workspace`](@ref); reuse across iterations with `ct_forward!` /
`ct_inverse!`.

!!! warning "Thread safety"
    A `CTWorkspace` is **not thread-safe**.  Create a separate workspace
    per thread (e.g., `[make_workspace(params, sz) for _ in 1:Threads.nthreads()]`).
"""
struct CTWorkspace{Td <: Number, Tf <: AbstractFloat, M <: AbstractMatrix{Td}, R}
    params::ContourletParams{Tf}     # real filter precision Tf
    image_size::Tuple{Int, Int}

    # Grow-once pool for all temporary buffers (LP coarse/bandpass, transients).
    # Separate arenas for forward and inverse passes because they traverse the
    # pyramid levels in opposite orders, yielding different buffer size sequences.
    fwd_scratch::ScratchArena{M}
    inv_scratch::ScratchArena{M}

    # Size of the bandpass image at each pyramid level.
    bp_dims::Vector{Tuple{Int, Int}}

    # DFB runners (equivalent filters, FFT buffers and plans) of the levels with a
    # directional split, found by bandpass size and depth (`_workspace_runner`).  A
    # level with L = 0 has none: the DFB just copies its bandpass there.
    dfb_runners::Vector{R}

    # Mutable slots for device-accelerated graph execs (e.g. CUDA.CuGraphExec).
    # Layout: [] = uncaptured; [exec, img_ptr::UInt, coarse_ptr::UInt].
    # Populated by ContourletsCUDAExt on first ct_forward!/ct_inverse! call.
    graph_cache::Vector{Any}
    inv_graph_cache::Vector{Any}
end

"""
    NSCTWorkspace{Td, Tf}

Precomputed masks, FFT plans and buffers (data type `Td`, real filter precision
`Tf`) for the nonsubsampled contourlet transform: the nonsubsampled pyramid as
masks on the input's spectrum and one nonsubsampled directional bank per level, so
`nsct_forward!` / `nsct_inverse!` are one FFT per output.  Create with
[`make_nsct_workspace`](@ref).

!!! warning "Thread safety"
    An `NSCTWorkspace` is **not thread-safe**; create one per concurrent caller.
    (A single call may still split its own subband FFTs between threads.)
"""
struct NSCTWorkspace{Td <: Number, Tf <: AbstractFloat, M <: AbstractMatrix{Td}, E <: _MaskBankEngine}
    params::ContourletParams{Tf}
    image_size::Tuple{Int, Int}

    # Staging for inputs / outputs that are not of the plans' dense type `M`.
    fwd_scratch::ScratchArena{M}
    inv_scratch::ScratchArena{M}

    # Pyramid masks (`_nsp_masks`: level j's bandpass and the lowpass), the NSDFB
    # bank of every level (the identity bank where L = 0), the outputs (j, k), and
    # the FFT plans and per-task buffers (`mask_bank.jl`).
    engine::E
end

Base.eltype(::CTWorkspace{Td}) where {Td} = Td
Base.eltype(::NSCTWorkspace{Td}) where {Td} = Td

# The dense matrix type `M` of a workspace.
_ws_matrix_type(::CTWorkspace{Td, Tf, M}) where {Td, Tf, M} = M
_ws_matrix_type(::NSCTWorkspace{Td, Tf, M}) where {Td, Tf, M} = M

# Bandpass size at each of the `J` pyramid levels of the CT (halving, rounded up).
function _bandpass_dims(image_size::Tuple{Int, Int}, J::Int)
    dims = Vector{Tuple{Int, Int}}(undef, J)
    d = image_size
    for j in 1:J
        dims[j] = d
        d = cld.(d, 2)
    end
    return dims
end

# The bank and runner types for arrays like `ref` at precision `Tf`, so that the
# workspace has one concrete type whatever the filter pair and depths (a tiny
# throwaway plan supplies the plan types).
_bank_type(ref::AbstractArray, ::Type{Tf}) where {Tf} =
    _DirectionalBank{typeof(similar(ref, Tf, 0, 0)), typeof(similar(ref, Complex{Tf}, 0))}

function _runner_type(ref::AbstractArray, ::Type{Tf}) where {Tf}
    C = typeof(similar(ref, Complex{Tf}, 0, 0))
    P, Q = typeof.(_inplace_plans(similar(ref, Complex{Tf}, 2, 2), FFTW.ESTIMATE))
    return _BankRunner{_bank_type(ref, Tf), C, P, Q}
end

# DFB runners of the split levels of a CT workspace.
function _dfb_level_runners(ref::AbstractMatrix, p::ContourletParams, bp_dims, ::Type{Tf}) where {Tf}
    flags = ref isa Array ? FFTW.MEASURE : FFTW.ESTIMATE
    pool = _runner_pool(ref, Tf)     # the levels run one after another
    return _runner_type(ref, Tf)[
        _bank_runner(
            _adapt_bank(ref, _directional_bank(Tf, dims, l, p.dfb_filters.f_ladder; decimated = true)),
            ref; flags = flags, pool = pool
        )
            for (l, dims) in zip(p.L_array, bp_dims) if l > 0
    ]
end

# The DFB runner a CT workspace holds for a bandpass of size `dims` split `l` deep
# with the ladder pair `qfp` at filter precision `Tf`, or `nothing` (also without a
# workspace, or when the workspace was built with other filters).
_workspace_runner(::Nothing, dims, l, half, qfp, Tf) = nothing

function _workspace_runner(
        ws::CTWorkspace{Td, Tw}, dims, l::Int, half::Symbol, qfp::QuincunxFilterPair, ::Type{Tf}
    ) where {Td, Tw, Tf}
    half === :both || return nothing
    (Tw === Tf && ws.params.dfb_filters.f_ladder == qfp.f_ladder) || return nothing
    for r in ws.dfb_runners
        (r.bank.dims == dims && r.bank.levels == l) && return r
    end
    return nothing
end

"""
    make_workspace(params::ContourletParams, image_size::Tuple{Int,Int}; T=Float64)
    make_workspace(T::Type, image_size::Tuple{Int,Int}, params::ContourletParams)
                   -> CTWorkspace

Allocate a [`CTWorkspace`](@ref) for images of `image_size` with the given `params`.
The second form is a convenience overload with the element type as the first positional argument.

# Examples
```jldoctest
julia> using Contourlets

julia> p = ContourletParams(J = 2, L_array = [1, 2]);

julia> ws = make_workspace(p, (64, 64));

julia> ws isa CTWorkspace
true
```
"""
make_workspace(
    T::Type{<:Number}, image_size::Tuple{Int, Int},
    params::ContourletParams; kwargs...
) =
    make_workspace(params, image_size; T = T, kwargs...)

"""
    make_workspace(image::AbstractMatrix, params::ContourletParams) -> CTWorkspace

Allocate a [`CTWorkspace`](@ref) using the array type and element type of `image`.
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
    bp_dims = _bandpass_dims(image_size, p2.J)
    ref = _allocate_zeros(M, Td, (1, 1))
    # The DFB equivalent filters, FFT buffers and plans of every level are built
    # once here (the allocating API rebuilds them per call).
    runners = _dfb_level_runners(ref, p2, bp_dims, Tf)
    ws = CTWorkspace{Td, Tf, M, eltype(runners)}(
        p2, image_size, ScratchArena{M}(), ScratchArena{M}(), bp_dims, runners, Any[], Any[]
    )
    # Grow the scratch arena to its steady-state size now (forward + inverse), so
    # the first user call is already allocation-free.
    prewarm && _prewarm_ct!(ws)
    return ws
end

"""
    make_nsct_workspace(params::ContourletParams, image_size::Tuple{Int,Int}; T=Float64,
                        threading=Auto()) -> NSCTWorkspace
    make_nsct_workspace(T::Type, image_size::Tuple{Int,Int}, params::ContourletParams)
                        -> NSCTWorkspace

Allocate an [`NSCTWorkspace`](@ref).  All buffers have the full `image_size`
since the NSCT performs no downsampling.
The second form is a convenience overload with the element type as the first positional argument.

!!! note "Threading"
    A host workspace holds one FFT buffer per Julia thread, and `nsct_forward!` /
    `nsct_inverse!` split their independent subband FFTs between threads when their
    own `threading` policy says so, call by call.  The `threading` kwarg here only
    applies to the warm-up pass.
"""
make_nsct_workspace(
    T::Type{<:Number}, image_size::Tuple{Int, Int},
    params::ContourletParams; kwargs...
) =
    make_nsct_workspace(params, image_size; T = T, kwargs...)

"""
    make_nsct_workspace(image::AbstractMatrix, params::ContourletParams) -> NSCTWorkspace

Allocate an [`NSCTWorkspace`](@ref) using the array type and element type of `image`.
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
    # Precompute the NSDFB equivalent filters (the bank depends on the pyramid level
    # only through its depth and its filter upsampling; levels with equal keys share
    # one bank, and depth 0 is the identity bank) and the pyramid masks; the `!`
    # paths then never rebuild them per call.
    ref = _allocate_zeros(M, Td, (1, 1))
    keys_j = [(p2.L_array[j], p2.L_array[j] == 0 ? 0 : _nsdfb_upsampling(j)) for j in 1:J]
    built = Dict{NTuple{2, Int}, _bank_type(ref, Tf)}()
    banks = map(keys_j) do (l, m)
        get!(built, (l, m)) do
            _adapt_bank(
                ref, _directional_bank(
                    Tf, image_size, l, p2.dfb_filters.f_ladder;
                    decimated = false, real_data = Td <: Real, upsample = m
                )
            )
        end
    end
    X, pfwd, pinv = _ns_plans(similar(ref, Td, image_size); measured = true)
    ana, syn = _nsp_masks(Tf, p2.lp_filters, image_size, J; rows = axes(X, 1))
    ana = [_like(X, a) for a in ana]
    syn = [_like(X, a) for a in syn]
    items = [_MaskItem((j, k), j, j, k) for j in 1:J for k in eachindex(banks[j].ana)]
    engine = _MaskBankEngine(ana[1:J], syn[1:J], ana[J + 1], syn[J + 1], banks, items, X, pfwd, pinv)
    ws = NSCTWorkspace{Td, Tf, M, typeof(engine)}(p2, image_size, ScratchArena{M}(), ScratchArena{M}(), engine)
    prewarm && _prewarm_nsct!(ws; threading = threading)
    return ws
end
