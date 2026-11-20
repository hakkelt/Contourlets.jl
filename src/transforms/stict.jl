# Semi-translation-invariant contourlet transform (STICT) of Eslami & Radha
# (2006): the translation-invariant pyramid followed by critically sampled DFBs.
# The coefficients are a `SubbandVector` (lowpass, then the subbands of each
# channel of each level); the directional banks write straight into its
# subbands.  The overview and the parts shared with the TICT are in
# `tilp_transforms.jl`.

"""
    STICTWorkspace

What an [`STICT`](@ref) precomputes for one image size, data element type and
array type: the translation-invariant pyramid's channel masks, the decimated
directional filter banks with their buffers, FFT buffers and plans.  Build it
with [`make_workspace`](@ref).

!!! warning "Thread safety"
    A workspace serves one call at a time; give each task its own.
"""
struct STICTWorkspace{Td <: Number, Tf <: AbstractFloat, C <: AbstractMatrix{Complex{Tf}}, R, P, Q, Tt <: STICT} <: AbstractTransformWorkspace
    transform::Tt
    params::STICT{Tf}
    image_size::NTuple{2, Int}
    layout::SubbandLayout{2}
    first::Vector{Int}                          # first subband of channel c of level i: first[4(i - 1) + c]
    masks::_TILPMasks{C}
    runners::Vector{Vector{R}}                  # [level][channel]; the identity bank when L = 0
    X::C                                        # full spectrum / synthesis accumulator
    Z::C                                        # scratch
    pfwd::P                                     # in-place FFT plans on X and Z
    pbwd::Q                                     # (inverse unnormalised)
    # Captured device graphs of the forward / inverse (`ContourletsCUDAExt`).
    graph_cache::_GraphCache
    inv_graph_cache::_GraphCache
end

MultiscaleTransformsBase.transform(ws::STICTWorkspace) = ws.transform
Base.size(ws::STICTWorkspace) = ws.image_size
Base.eltype(::Type{<:STICTWorkspace{Td}}) where {Td} = Td
Base.show(io::IO, ws::STICTWorkspace) = print(io, "STICTWorkspace(", ws.transform, ", ", eltype(ws), ", ", ws.image_size, ")")

# The subband layout of an STICT on a `dims` image, and the index of the first
# subband of each channel.
function _layout(t::STICT, dims::NTuple{2, Int})
    n1, n2 = dims
    groups = [dfb_subband_sizes(n1, n2, l; half = h) for l in t.L_array for h in _TILP_HALVES]
    return _stack_levels(dims, groups, dims)
end

function MultiscaleTransformsBase.make_workspace(
        t::STICT, ::Type{T}, dims::Dims; prewarm::Bool = true,
        threaded::Union{Bool, Nothing} = nothing, tune::Bool = true
    ) where {T <: Number}
    Td = data_eltype(T)
    return _stict_workspace(t, Td, _check_image(dims), Matrix{Td}, prewarm)
end

function MultiscaleTransformsBase.make_workspace(
        t::STICT, x::AbstractArray; prewarm::Bool = true,
        threaded::Union{Bool, Nothing} = nothing, tune::Bool = true
    )
    Td = data_eltype(x)
    return _stict_workspace(t, Td, _check_image(size(x)), _matrix_type(x, Td), prewarm)
end

function _stict_workspace(t::STICT, ::Type{Td}, image_size::NTuple{2, Int}, ::Type{M}, prewarm::Bool) where {Td, M}
    Tf = filter_eltype(Td)
    p = _convert_transform(Tf, t)
    layout, firsts = _layout(p, image_size)
    ref = _allocate_zeros(M, Td, image_size)
    masks = _adapt_masks(ref, _tilp_masks(Tf, p.lp_filters, image_size, p.J))
    R = _runner_type(ref, Tf)
    # One full bank per depth, the half trees sharing its masks (`_half_bank`), and
    # one buffer pool for every runner: the channels run one after another.  The
    # runners' scratch `Z` is also the workspace's, which is never live meanwhile.
    flags = ref isa Array ? FFTW.MEASURE : FFTW.ESTIMATE
    banks = Dict{Int, _bank_type(ref, Tf)}()
    full(l) = get!(banks, l) do
        _adapt_bank(ref, _directional_bank(Tf, image_size, l, p.dfb_filters.f_ladder; decimated = true))
    end
    # A level with L = 0 runs its channels through the identity bank: one
    # undecimated channel with unit masks and phases, so the same FFT count as
    # storing the channel directly, plus a few elementwise passes and one extra
    # full-size buffer.
    pool = _runner_pool(ref, Tf)
    built = Dict{Tuple{Int, Symbol}, R}()
    runner(l, half) = get!(built, (l, l == 0 ? :both : half)) do
        bank = l == 0 ? full(0) : _half_bank(full(l), full(1), half)
        _bank_runner(bank, ref; flags = flags, pool = pool)
    end
    runners = [R[runner(l, h) for h in _TILP_HALVES] for l in p.L_array]
    X = fill!(similar(ref, Complex{Tf}, image_size), zero(Complex{Tf}))
    pfwd, pbwd = _inplace_plans(X, flags)
    Z = first(pool[(:Z, image_size)])
    ws = STICTWorkspace{Td, Tf, typeof(X), R, typeof(pfwd), typeof(pbwd), typeof(t)}(
        t, p, image_size, layout, firsts, masks, runners, X, Z, pfwd, pbwd, _GraphCache(), _GraphCache()
    )
    if prewarm
        c = SubbandVector(fill!(similar(ref, Td, length(layout)), zero(Td)), layout)
        _stict_forward_inner!(c, ref, ws)
        _stict_inverse_inner!(ref, c, ws)
    end
    return ws
end

# ── Transforms ────────────────────────────────────────────────────────────────

# Without a workspace: one is built for the call.
function _analyze_direct!(c::SubbandVector, first::Vector{Int}, t::STICT, x::AbstractMatrix)
    Td = eltype(c)
    ws = _stict_workspace(t, Td, size(x), _matrix_type(x, Td), false)
    return _stict_forward_inner!(c, Td === eltype(x) ? x : Td.(x), ws)
end

function _synthesize_direct!(x::AbstractMatrix, first::Vector{Int}, t::STICT, c::SubbandVector)
    Td = eltype(c)
    ws = _stict_workspace(t, Td, size(x), _matrix_type(c.data, Td), false)
    Td === eltype(x) && return _stict_inverse_inner!(x, c, ws)
    return copyto!(x, _stict_inverse_inner!(similar(c.data, Td, size(x)), c, ws))
end

# Typed dispatch barriers; `capture` selects the seam that `ContourletsCUDAExt`
# overloads to capture device graphs.
function _ragged_forward!(c::SubbandVector, x::AbstractMatrix, ws::STICTWorkspace, capture::Bool)
    T = eltype(ws)
    img = T === eltype(x) ? x : T.(x)
    return capture ? _stict_forward_ws!(c, img, ws) : _stict_forward_inner!(c, img, ws)
end

_ragged_inverse!(x::AbstractMatrix, c::SubbandVector, ws::STICTWorkspace, capture::Bool) =
    capture ? _stict_inverse_ws!(x, c, ws) : _stict_inverse_inner!(x, c, ws)

# The workspace paths: the same kernels on the same buffers every call, which
# `ContourletsCUDAExt` captures as CUDA graphs.  Nothing in them may allocate.
_stict_forward_ws!(coeffs::SubbandVector, x::AbstractMatrix, ws::STICTWorkspace) = _stict_forward_inner!(coeffs, x, ws)
_stict_inverse_ws!(x::AbstractMatrix, coeffs::SubbandVector, ws::STICTWorkspace) = _stict_inverse_inner!(x, coeffs, ws)

function _stict_forward_inner!(coeffs::SubbandVector, x::AbstractMatrix, ws::STICTWorkspace)
    m, X, Z = ws.masks, ws.X, ws.Z
    X .= x
    ws.pfwd * X
    for i in eachindex(m.ana), c in 1:4
        g = 4 * (i - 1) + c
        r = ws.runners[i][c]
        out = _Bands(coeffs, ws.first[g], _tict_n_subbands(ws.params.L_array[i], c))
        R, K = r.X, m.ana[i][c]
        @.. R = K * X               # FastBroadcast: see `_mask_bank_forward_items!`
        _bank_analyze_spectrum!(out, r)
    end
    Kc = m.coarse_ana
    @.. Z = Kc * X
    ws.pbwd * Z
    _store!(subband(coeffs, 1), Z, _bwd_scale(Z))
    return coeffs
end

function _stict_inverse_inner!(x::AbstractMatrix, coeffs::SubbandVector, ws::STICTWorkspace)
    m, X, Z = ws.masks, ws.X, ws.Z
    fill!(X, zero(eltype(X)))
    for i in eachindex(m.syn), c in 1:4
        r = ws.runners[i][c]
        sbs = _Bands(coeffs, ws.first[4 * (i - 1) + c], _tict_n_subbands(ws.params.L_array[i], c))
        _bank_synthesize_spectrum!(sbs, r)
        R, K = r.X, m.syn[i][c]
        @.. X += K * R
    end
    Z .= subband(coeffs, 1)
    ws.pfwd * Z
    Kc = m.coarse_syn
    @.. X += Kc * Z
    ws.pbwd * X
    return _store!(x, X, _bwd_scale(X))
end
