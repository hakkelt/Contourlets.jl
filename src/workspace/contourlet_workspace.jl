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
    first::Vector{Int}               # the first subband of each scale

    # Grow-once pool for all temporary buffers (LP coarse/bandpass, transients).
    # Separate arenas for forward and inverse passes because they traverse the
    # pyramid levels in opposite orders, yielding different buffer size sequences.
    fwd_scratch::ScratchArena{M}
    inv_scratch::ScratchArena{M}

    # Size of the bandpass image at each scale.
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

    # Pyramid masks (`_nsp_masks`: the bandpass of each scale and the lowpass), the
    # NSDFB bank of every scale (the identity bank where L = 0), the outputs (j, k), and
    # the FFT plans and per-task buffers (`mask_bank.jl`).
    engine::E
end

const _AnyWorkspace = Union{CTWorkspace, NSCTWorkspace}

AbstractMultiscaleTransforms.transform(ws::_AnyWorkspace) = ws.transform
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

# Bandpass size at each of the `J` scales of the CT, coarsest first: the finest
# has the image's size, and each coarser one half of it (rounded up).
function _bandpass_dims(image_size::Tuple{Int, Int}, J::Int)
    dims = Vector{Tuple{Int, Int}}(undef, J)
    d = image_size
    for j in J:-1:1
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
# `flags` are the FFTW planner flags of host plans.
function _dfb_level_runners(ref::AbstractMatrix, p::CT, bp_dims, ::Type{Tf}, flags::Integer) where {Tf}
    pool = _runner_pool(ref, Tf)     # the levels run one after another
    return _runner_type(ref, Tf)[
        _bank_runner(
            _adapt_bank(ref, _directional_bank(Tf, dims, l, p.dfb_filters.f_ladder; decimated = true)),
            ref; flags = flags, pool = pool
        )
            for (l, dims) in zip(p.direction_levels, bp_dims) if l > 0
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

# `tasks` of `make_workspace`: `nothing` or at least 1.
function _check_tasks(tasks::Union{Integer, Nothing})
    tasks === nothing || tasks >= 1 || throw(ArgumentError("tasks must be ≥ 1, got $(tasks)"))
    return tasks
end

const _WORKSPACE_KEYWORDS = """
Keywords, accepted by the workspaces of all four transforms (the shared
keywords of `make_workspace` in AbstractMultiscaleTransforms):
- `flags = FFTW.MEASURE`: the FFTW planner flags of the host FFT plans — those
  of the directional filter banks of a [`CT`](@ref) or [`STICT`](@ref), and
  the image-size FFTs of an [`NSCT`](@ref) or [`TICT`](@ref), which are
  planned single-threaded because the workspace threads over its subbands
  instead.  `FFTW.ESTIMATE` builds the workspace faster, possibly with slower
  plans.  Device plans ignore it.
- `tasks = nothing`: the most tasks an [`NSCT`](@ref) or [`TICT`](@ref) host
  workspace splits its subband FFTs between, and so the number of its per-task
  FFT buffers; `nothing` means one per Julia thread, `1` a serial workspace
  whose calls run on the calling task whatever their `threaded`.  It must be
  at least 1.  The [`CT`](@ref) and [`STICT`](@ref) run serially and a
  device workspace runs one task, so there it is checked and otherwise ignored.
- `tune = false`: for [`NSCT`](@ref) and [`TICT`](@ref) workspaces on a device,
  measure how many subbands each batched FFT should take, and whether the
  synthesis should be batched too (`MultiscaleFilterBanks.mask_bank_tune`); this
  runs a few transforms at construction.  On an A100 (Float32, 256² and 512²)
  tuning adds 40–240 ms to the construction and makes the synthesis 1.6–2.6×
  and the TICT analysis 15–25% faster, so it pays off for a workspace that runs
  more than about 150 (NSCT) to 330 (TICT) analysis–synthesis pairs, as in an
  iterative reconstruction.  Host workspaces thread over their subbands
  instead, and the [`CT`](@ref) and [`STICT`](@ref) have nothing to tune:
  there it is ignored.
- `prewarm = true`: run one analysis and synthesis on zeros at construction, so
  that the first call is compiled and allocates nothing.  The warm-up threads
  as a call with the default `threaded = nothing` does."""

"""
    make_workspace(t::Union{CT, NSCT, TICT, STICT}, T::Type{<:Number}, dims; flags = FFTW.MEASURE, tasks = nothing, tune = false, prewarm = true)
    make_workspace(t::Union{CT, NSCT, TICT, STICT}, x::AbstractMatrix; kwargs...)

The workspace of the contourlet transform `t` ([`CTWorkspace`](@ref),
[`NSCTWorkspace`](@ref), [`TICTWorkspace`](@ref) or [`STICTWorkspace`](@ref))
for images of size `dims` and element type `T` (real or complex; an integer
type means `float(T)`), on the host.  The second form takes size, element type
and array type from the image `x`; pass a device array to build a device
workspace.  With it, `analyze!` and `synthesize!` allocate nothing.  The
workspace runs at the precision of `T`: a `Float32` workspace converts the
filters of a `Float64` transform.  A workspace's buffers serve one call at a
time: tasks that run transforms concurrently need a workspace each.

$(_WORKSPACE_KEYWORDS)

# Examples
```jldoctest
julia> using Contourlets

julia> t = CT(scales = 2, direction_levels = [1, 2]);

julia> ws = make_workspace(t, Float64, (64, 64));

julia> transform(ws) == t, eltype(ws), size(ws)
(true, Float64, (64, 64))
```
"""
function AbstractMultiscaleTransforms.make_workspace(
        t::CT, ::Type{T}, dims::Dims; flags::Integer = FFTW.MEASURE,
        tasks::Union{Integer, Nothing} = nothing, tune::Bool = false, prewarm::Bool = true
    ) where {T <: Number}
    Td = data_eltype(T)
    _check_tasks(tasks)
    return _ct_workspace(t, Td, _check_image(dims), Matrix{Td}, prewarm, flags)
end

function AbstractMultiscaleTransforms.make_workspace(
        t::CT, x::AbstractArray; flags::Integer = FFTW.MEASURE,
        tasks::Union{Integer, Nothing} = nothing, tune::Bool = false, prewarm::Bool = true
    )
    Base.require_one_based_indexing(x)
    Td = data_eltype(x)
    _check_tasks(tasks)
    return _ct_workspace(t, Td, _check_image(size(x)), _matrix_type(x, Td), prewarm, flags)
end

function _ct_workspace(t::CT, ::Type{Td}, image_size::NTuple{2, Int}, ::Type{M}, prewarm::Bool, flags::Integer) where {Td, M}
    Tf = filter_eltype(Td)          # real filter precision
    p2 = _convert_transform(Tf, t)
    bp_dims = _bandpass_dims(image_size, p2.scales)
    layout, firsts = _layout(p2, image_size)
    ref = _allocate_zeros(M, Td, (1, 1))
    # The DFB equivalent filters, FFT buffers and plans of every level are built
    # once here (the allocating API rebuilds them per call).
    runners = _dfb_level_runners(ref, p2, bp_dims, Tf, flags)
    ws = CTWorkspace{Td, Tf, M, eltype(runners), typeof(t)}(
        t, p2, image_size, layout, firsts, ScratchArena{M}(), ScratchArena{M}(), bp_dims, runners,
        _GraphCache(), _GraphCache()
    )
    # Grow the scratch arena to its steady-state size now (forward + inverse), so
    # the first user call is already allocation-free.
    prewarm && _prewarm_ct!(ws)
    return ws
end

function AbstractMultiscaleTransforms.make_workspace(
        t::NSCT, ::Type{T}, dims::Dims; flags::Integer = FFTW.MEASURE,
        tasks::Union{Integer, Nothing} = nothing, tune::Bool = false, prewarm::Bool = true
    ) where {T <: Number}
    Td = data_eltype(T)
    return _nsct_workspace(t, Td, _check_image(dims), Matrix{Td}, prewarm, flags, _check_tasks(tasks), tune)
end

function AbstractMultiscaleTransforms.make_workspace(
        t::NSCT, x::AbstractArray; flags::Integer = FFTW.MEASURE,
        tasks::Union{Integer, Nothing} = nothing, tune::Bool = false, prewarm::Bool = true
    )
    Base.require_one_based_indexing(x)
    Td = data_eltype(x)
    return _nsct_workspace(t, Td, _check_image(size(x)), _matrix_type(x, Td), prewarm, flags, _check_tasks(tasks), tune)
end

function _nsct_workspace(
        t::NSCT, ::Type{Td}, image_size::NTuple{2, Int}, ::Type{M}, prewarm::Bool, flags::Integer,
        tasks::Union{Integer, Nothing}, tune::Bool
    ) where {Td, M}
    Tf = filter_eltype(Td)          # real filter precision
    p2 = _convert_transform(Tf, t)
    J = p2.scales
    # Precompute the NSDFB equivalent filters (the bank depends on the scale only
    # through its depth and its filter upsampling; scales with equal keys share
    # one bank, and depth 0 is the identity bank) and the pyramid masks; the `!`
    # paths then never rebuild them per call.
    ref = _allocate_zeros(M, Td, (1, 1))
    # Scale j is the bandpass of the pyramid's split J + 1 - j (1 = the finest).
    keys_j = [(l, l == 0 ? 0 : _nsdfb_upsampling(J + 1 - j)) for (j, l) in enumerate(p2.direction_levels)]
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
    X, pfwd, pbwd = _ns_unnormalized_plans(x; flags, unaligned = _unaligned_slices(x))
    ana, syn = _nsp_masks(Tf, p2.lp_filters, image_size, J; rows = axes(X, 1))
    # By scale, coarsest first; the lowpass stays last.
    ana = [_like(X, a) for a in ana[[J:-1:1; J + 1]]]
    syn = [_like(X, a) for a in syn[[J:-1:1; J + 1]]]
    items = [MaskItem((j, k), j, j, k) for j in 1:J for k in eachindex(banks[j].ana)]
    offsets = _nsct_offsets(p2)
    n = nsubbands(p2, image_size)
    stack = similar(ref, Td, (0, 0, 0))
    engine = _mask_bank_engine(x, X, pfwd, pbwd, ana[1:J], syn[1:J], ana[J + 1], syn[J + 1], banks, items; tasks, tunable = tune)
    engine = _maybe_tune(engine, tune, tune ? _Stacked(_zero_stack(ref, Td, image_size, n), offsets) : nothing, x)
    ws = NSCTWorkspace{Td, Tf, M, typeof(engine), typeof(stack), typeof(t)}(
        t, p2, image_size, offsets, n, stack, ScratchArena{M}(), ScratchArena{M}(), engine
    )
    prewarm && _prewarm_stacked!(ws)
    return ws
end
