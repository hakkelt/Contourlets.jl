# CTWorkspace and NSCTWorkspace: what the CT and NSCT precompute for one image
# size and data element type (filters, FFT buffers and plans, scratch buffers),
# so that `analyze!` / `synthesize!` with a workspace allocate nothing.  A
# workspace is not safe for concurrent use: give each task its own.
#
# Scratch buffers come from the `ScratchArena` in `scratch_arena.jl`.

"""
    CTWorkspace

What a [`CT`](@ref) precomputes for one image size, data element type and array
type: the pyramid's scratch arenas and, per pyramid level with a directional
split, the DFB's equivalent filters, FFT buffers and plans.  Build it with
[`make_workspace`](@ref); `transform`, `eltype` and `size` give the transform,
data element type and image size it serves.

!!! warning "Thread safety"
    A workspace serves one call at a time; give each task its own.
"""
struct CTWorkspace{Td <: Number, Tf <: AbstractFloat, M <: AbstractMatrix{Td}, R, Tt <: CT} <: AbstractTransformWorkspace
    transform::Tt                    # as given to `make_workspace`
    params::CT{Tf}                   # the transform at filter precision Tf
    image_size::Tuple{Int, Int}
    layout::SubbandLayout{2}
    first::Vector{Int}               # the first subband of each level

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

    # Captured device graphs of the forward / inverse (`ContourletsCUDAExt`).
    graph_cache::_GraphCache
    inv_graph_cache::_GraphCache
end

"""
    NSCTWorkspace

What an [`NSCT`](@ref) precomputes for one image size, data element type and
array type: the nonsubsampled pyramid as masks on the image's spectrum, one
nonsubsampled directional bank per level, FFT plans and buffers, so that
`analyze!` / `synthesize!` are one FFT per subband.  Build it with
[`make_workspace`](@ref).

!!! warning "Thread safety"
    A workspace serves one call at a time; give each task its own.  (A single
    call may still split its own subband FFTs between threads.)
"""
struct NSCTWorkspace{Td <: Number, Tf <: AbstractFloat, M <: AbstractMatrix{Td}, E <: MaskBankEngine, S <: AbstractArray{Td, 3}, Tt <: NSCT} <: AbstractTransformWorkspace
    transform::Tt
    params::NSCT{Tf}
    image_size::Tuple{Int, Int}
    # Subband (j, k) is slice `offsets[j] + k` of the coefficients.
    offsets::Vector{Int}
    nsubbands::Int
    # An empty array of the coefficient type the engine writes into directly.
    stack::S

    # Staging for inputs / outputs that are not of the plans' dense type `M`.
    fwd_scratch::ScratchArena{M}
    inv_scratch::ScratchArena{M}

    # Pyramid masks (`_nsp_masks`: level j's bandpass and the lowpass), the NSDFB
    # bank of every level (the identity bank where L = 0), the outputs (j, k), and
    # the FFT plans and per-task buffers (`mask_bank.jl`).
    engine::E
end

const _AnyWorkspace = Union{CTWorkspace, NSCTWorkspace}

MultiscaleTransformsBase.transform(ws::_AnyWorkspace) = ws.transform
Base.size(ws::_AnyWorkspace) = ws.image_size
Base.eltype(::Type{<:CTWorkspace{Td}}) where {Td} = Td
Base.eltype(::Type{<:NSCTWorkspace{Td}}) where {Td} = Td

function Base.show(io::IO, ws::_AnyWorkspace)
    return print(io, nameof(typeof(ws)), "(", ws.transform, ", ", eltype(ws), ", ", ws.image_size, ")")
end

# The dense matrix type `M` of a workspace.
_ws_matrix_type(::CTWorkspace{Td, Tf, M}) where {Td, Tf, M} = M
_ws_matrix_type(::NSCTWorkspace{Td, Tf, M}) where {Td, Tf, M} = M

# The dense matrix type of data like `x` with element type `Td` (host or device).
_matrix_type(x::AbstractArray, ::Type{Td}) where {Td} = typeof(similar(x, Td, 0, 0))

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
function _dfb_level_runners(ref::AbstractMatrix, p::CT, bp_dims, ::Type{Tf}) where {Tf}
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
# with the ladder pair `qfp` at filter precision `Tf`; `nothing` without a
# workspace.  A workspace that cannot serve the call throws.
_workspace_runner(::Nothing, dims, l, half, qfp, Tf) = nothing

function _workspace_runner(ws, dims, l, half, qfp, Tf)
    throw(ArgumentError("workspace must be a CTWorkspace or nothing, got $(typeof(ws))"))
end

function _workspace_runner(
        ws::CTWorkspace{Td, Tw}, dims, l::Int, half::Symbol, qfp::QuincunxFilterPair, ::Type{Tf}
    ) where {Td, Tw, Tf}
    half === :both || throw(
        ArgumentError("a CTWorkspace holds full-tree DFBs only; pass workspace = nothing for half = :$(half)")
    )
    Tw === Tf || throw(
        ArgumentError("the workspace was built for filter precision $(Tw), but the data needs $(Tf)")
    )
    ws.params.dfb_filters == qfp || throw(
        ArgumentError("the workspace was built with different DFB filters than the ones passed")
    )
    for r in ws.dfb_runners
        (r.bank.dims == dims && r.bank.levels == l) && return r
    end
    held = [(r.bank.dims, r.bank.levels) for r in ws.dfb_runners]
    throw(
        ArgumentError(
            "the workspace (image size $(ws.image_size)) has no DFB for a $(dims) bandpass at depth $(l); it holds (size, depth) = $(held)"
        )
    )
end

const _WORKSPACE_KEYWORDS = """
Keywords, accepted by the workspaces of all four transforms:
- `prewarm = true`: run one analysis and synthesis on zeros at construction, so
  that the first call is compiled and allocates nothing;
- `threaded = nothing`: threading of that warm-up pass (the calls choose their
  own threading);
- `tune = false`: for [`NSCT`](@ref) and [`TICT`](@ref) workspaces on a device,
  measure how many subbands each batched FFT should take, and whether the
  synthesis should be batched too (`MultiscaleFilterBanks.tune_mask_bank`); this
  runs a few transforms at construction.  Host workspaces thread over their
  subbands instead, and the [`CT`](@ref) and [`STICT`](@ref) have nothing to
  tune: there it does nothing."""

"""
    make_workspace(t::Union{CT, NSCT, TICT, STICT}, T::Type{<:Number}, dims; prewarm = true, threaded = nothing, tune = false)
    make_workspace(t::Union{CT, NSCT, TICT, STICT}, x::AbstractMatrix; kwargs...)

The workspace of the contourlet transform `t` ([`CTWorkspace`](@ref),
[`NSCTWorkspace`](@ref), [`TICTWorkspace`](@ref) or [`STICTWorkspace`](@ref))
for images of size `dims` and element type `T` (real or complex; an integer
type means `float(T)`), on the host.  The second form takes size, element type
and array type from the image `x`; pass a device array to build a device
workspace.  With it, `analyze!` and `synthesize!` allocate nothing.  The
workspace runs at the precision of `T`: a `Float32` workspace converts the
filters of a `Float64` transform.

$(_WORKSPACE_KEYWORDS)

# Examples
```jldoctest
julia> using Contourlets

julia> t = CT(J = 2, L_array = [1, 2]);

julia> ws = make_workspace(t, Float64, (64, 64));

julia> transform(ws) == t, eltype(ws), size(ws)
(true, Float64, (64, 64))
```
"""
function MultiscaleTransformsBase.make_workspace(
        t::CT, ::Type{T}, dims::Dims; prewarm::Bool = true,
        threaded::Union{Bool, Nothing} = nothing, tune::Bool = false
    ) where {T <: Number}
    Td = data_eltype(T)
    return _ct_workspace(t, Td, _check_image(dims), Matrix{Td}, prewarm)
end

function MultiscaleTransformsBase.make_workspace(
        t::CT, x::AbstractArray; prewarm::Bool = true,
        threaded::Union{Bool, Nothing} = nothing, tune::Bool = false
    )
    Td = data_eltype(x)
    return _ct_workspace(t, Td, _check_image(size(x)), _matrix_type(x, Td), prewarm)
end

function _ct_workspace(t::CT, ::Type{Td}, image_size::NTuple{2, Int}, ::Type{M}, prewarm::Bool) where {Td, M}
    Tf = filter_eltype(Td)          # real filter precision
    p2 = _convert_transform(Tf, t)
    bp_dims = _bandpass_dims(image_size, p2.J)
    layout, firsts = _layout(p2, image_size)
    ref = _allocate_zeros(M, Td, (1, 1))
    # The DFB equivalent filters, FFT buffers and plans of every level are built
    # once here (the allocating API rebuilds them per call).
    runners = _dfb_level_runners(ref, p2, bp_dims, Tf)
    ws = CTWorkspace{Td, Tf, M, eltype(runners), typeof(t)}(
        t, p2, image_size, layout, firsts, ScratchArena{M}(), ScratchArena{M}(), bp_dims, runners,
        _GraphCache(), _GraphCache()
    )
    # Grow the scratch arena to its steady-state size now (forward + inverse), so
    # the first user call is already allocation-free.
    prewarm && _prewarm_ct!(ws)
    return ws
end

function MultiscaleTransformsBase.make_workspace(
        t::NSCT, ::Type{T}, dims::Dims; prewarm::Bool = true,
        threaded::Union{Bool, Nothing} = nothing, tune::Bool = false
    ) where {T <: Number}
    Td = data_eltype(T)
    return _nsct_workspace(t, Td, _check_image(dims), Matrix{Td}, prewarm, threaded, tune)
end

function MultiscaleTransformsBase.make_workspace(
        t::NSCT, x::AbstractArray; prewarm::Bool = true,
        threaded::Union{Bool, Nothing} = nothing, tune::Bool = false
    )
    Td = data_eltype(x)
    return _nsct_workspace(t, Td, _check_image(size(x)), _matrix_type(x, Td), prewarm, threaded, tune)
end

function _nsct_workspace(
        t::NSCT, ::Type{Td}, image_size::NTuple{2, Int}, ::Type{M}, prewarm::Bool,
        threaded::Union{Bool, Nothing}, tune::Bool
    ) where {Td, M}
    Tf = filter_eltype(Td)          # real filter precision
    p2 = _convert_transform(Tf, t)
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
    x = similar(ref, Td, image_size)
    X, pfwd, pbwd = _ns_unnormalised_plans(x; measured = true, unaligned = _unaligned_slices(x))
    ana, syn = _nsp_masks(Tf, p2.lp_filters, image_size, J; rows = axes(X, 1))
    ana = [_like(X, a) for a in ana]
    syn = [_like(X, a) for a in syn]
    items = [MaskItem((j, k), j, j, k) for j in 1:J for k in eachindex(banks[j].ana)]
    offsets = _nsct_offsets(p2)
    n = nsubbands(p2, image_size)
    stack = similar(ref, Td, (0, 0, 0))
    engine = _mask_bank_engine(x, X, pfwd, pbwd, ana[1:J], syn[1:J], ana[J + 1], syn[J + 1], banks, items; tunable = tune)
    engine = _maybe_tune(engine, tune, tune ? _Stacked(_zero_stack(ref, Td, image_size, n), offsets) : nothing, x)
    ws = NSCTWorkspace{Td, Tf, M, typeof(engine), typeof(stack), typeof(t)}(
        t, p2, image_size, offsets, n, stack, ScratchArena{M}(), ScratchArena{M}(), engine
    )
    prewarm && _prewarm_stacked!(ws, threaded)
    return ws
end
