# Translation-invariant contourlet transform (TICT) of Eslami & Radha (2006): the
# translation-invariant pyramid followed by nonsubsampled DFBs, on the DFT grid.
# The workspace runs the mask-bank engine on stacked coefficients, as the NSCT's
# does (`common.jl`); the overview and the parts shared with the STICT are in
# `tilp_transforms.jl`.

"""
    TICTWorkspace

What a [`TICT`](@ref) precomputes for one image size, data element type and
array type: the translation-invariant pyramid's channel masks, the directional
filters, FFT buffers and plans.  Build it with [`make_workspace`](@ref).

!!! warning "Thread safety"
    A workspace serves one call at a time; give each task its own.
"""
struct TICTWorkspace{Td <: Number, Tf <: AbstractFloat, M <: AbstractMatrix, E <: MaskBankEngine, S <: AbstractArray{Td, 3}, Tt <: TICT} <: AbstractTransformWorkspace
    transform::Tt
    params::TICT{Tf}
    image_size::NTuple{2, Int}
    # Subband (i, c, k) is slice `offsets[4(i - 1) + c] + k` of the coefficients.
    offsets::Vector{Int}
    nsubbands::Int
    # An empty array of the coefficient type the engine writes into directly.
    stack::S
    # Staging buffers for an input or output that is not of the dense type `M`
    # the FFT plans run on (a view, a transpose; `_dense_input`).
    fwd_scratch::ScratchArena{M}
    inv_scratch::ScratchArena{M}
    # TILP channel masks (pyramid mask 4(i - 1) + c is channel c of level i) and
    # lowpass, the directional bank of every channel (bank index = pyramid index;
    # the identity bank where L = 0), the outputs (i, c, k), and the FFT plans and
    # per-task buffers (`mask_bank.jl`).
    engine::E
end

MultiscaleTransformsBase.transform(ws::TICTWorkspace) = ws.transform
Base.size(ws::TICTWorkspace) = ws.image_size
Base.eltype(::Type{<:TICTWorkspace{Td}}) where {Td} = Td
_ws_matrix_type(::TICTWorkspace{Td, Tf, M}) where {Td, Tf, M} = M
Base.show(io::IO, ws::TICTWorkspace) = print(io, "TICTWorkspace(", ws.transform, ", ", eltype(ws), ", ", ws.image_size, ")")

function MultiscaleTransformsBase.make_workspace(
        t::TICT, ::Type{T}, dims::Dims; prewarm::Bool = true,
        threaded::Union{Bool, Nothing} = nothing, tune::Bool = false
    ) where {T <: Number}
    Td = data_eltype(T)
    return _tict_workspace(t, Td, _check_image(dims), Matrix{Td}, prewarm, threaded, tune)
end

function MultiscaleTransformsBase.make_workspace(
        t::TICT, x::AbstractArray; prewarm::Bool = true,
        threaded::Union{Bool, Nothing} = nothing, tune::Bool = false
    )
    Td = data_eltype(x)
    return _tict_workspace(t, Td, _check_image(size(x)), _matrix_type(x, Td), prewarm, threaded, tune)
end

function _tict_workspace(
        t::TICT, ::Type{Td}, image_size::NTuple{2, Int}, ::Type{M}, prewarm::Bool,
        threaded::Union{Bool, Nothing}, tune::Bool
    ) where {Td, M}
    Tf = filter_eltype(Td)
    p = _convert_transform(Tf, t)
    n1, n2 = image_size
    real_data = Td <: Real
    ref = _allocate_zeros(M, Td, image_size)
    rows = real_data ? (1:(n1 ÷ 2 + 1)) : (1:n1)
    masks = _adapt_masks(ref, _tilp_masks(Tf, p.lp_filters, image_size, p.J; rows = rows))
    B = _bank_type(ref, Tf)
    # One full bank per depth (the identity bank at depth 0); the half trees share
    # its masks (`_half_bank`).
    built = Dict{Int, B}()
    full(l) = get!(built, l) do
        _adapt_bank(ref, _directional_bank(Tf, image_size, l, p.dfb_filters.f_ladder; decimated = false, real_data = real_data))
    end
    bank(l, half) = l == 0 ? full(0) : _half_bank(full(l), full(1), half)
    banks = B[bank(l, h) for l in p.L_array for h in _TILP_HALVES]
    g(i, c) = 4 * (i - 1) + c                   # pyramid mask and bank of channel c of level i
    items = [
        MaskItem((i, c, k), g(i, c), g(i, c), k)
            for (i, l) in enumerate(p.L_array) for c in 1:4 for k in 1:_tict_n_subbands(l, c)
    ]
    X, pfwd, pbwd = _ns_unnormalised_plans(ref; measured = true, unaligned = _unaligned_slices(ref))
    engine = _mask_bank_engine(
        ref, X, pfwd, pbwd, reduce(vcat, masks.ana), reduce(vcat, masks.syn), masks.coarse_ana, masks.coarse_syn,
        banks, items; tunable = tune
    )
    offsets = _tilp_offsets(p)
    n = nsubbands(p, image_size)
    engine = _maybe_tune(engine, tune, tune ? _Stacked(_zero_stack(ref, Td, image_size, n), offsets) : nothing, ref)
    Mw = typeof(ref)
    stack = similar(ref, Td, (0, 0, 0))
    ws = TICTWorkspace{Td, Tf, Mw, typeof(engine), typeof(stack), typeof(t)}(
        t, p, (n1, n2), offsets, n, stack, ScratchArena{Mw}(), ScratchArena{Mw}(), engine
    )
    prewarm && _prewarm_stacked!(ws, threaded)
    return ws
end

# Without a workspace: one is built for the call.
function _analyze_direct!(c::AbstractArray, t::TICT, x::AbstractMatrix, threaded)
    Td = eltype(c)
    ws = _tict_workspace(t, Td, size(x), _matrix_type(x, Td), false, nothing, false)
    return _stacked_forward!(c, x, ws, threaded)
end

function _synthesize_direct!(x::AbstractMatrix, t::TICT, c::AbstractArray, threaded)
    Td = eltype(c)
    ws = _tict_workspace(t, Td, size(x), _matrix_type(c, Td), false, nothing, false)
    return _stacked_inverse!(x, c, ws, threaded)
end
