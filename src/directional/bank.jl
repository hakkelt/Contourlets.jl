# The runtime directional bank: `_DirectionalBank` (the equivalent filters of
# `bank_design.jl` sampled for one image size, in filter precision), the
# `_directional_bank` constructor with its argument checks, and the quincunx
# storage of depth-1 channels.

# ── Runtime bank ──────────────────────────────────────────────────────────────

"""
    _DirectionalBank

The equivalent filters of one directional bank for one image size, ready to apply:
real amplitude masks in filter precision, stored as host or device arrays.

* Decimated (DFB): channel `k` computes `yₖ[n] = (hₖ ⊛ x)[Sₖn]` with
  `Hₖ(ω) = anaₖ(ω) e^{jω·dₖ}`.  The fold over the `|det Sₖ|` aliases is normalised
  inside `ana`, so synthesis is `x = Σₖ gₖ ⊛ (↑Sₖ yₖ)` with `Gₖ = synₖ e^{−jω·dₖ}`.
  Depth-1 channels (`l = 1`, or the unsplit fan of a half tree) are sampled on the
  quincunx lattice and stored column by column (see `_quincunx_pack!`).
* Nonsubsampled (NSDFB): `yₖ = hₖ ⊛ x` with the zero-phase `Hₖ = anaₖ`, both
  amplitudes scaled by `2^(-depth/2)` so that `Σₖ GₖHₖ = 1`.  The delays only move
  the decimation lattice, so a bank without decimation drops them and every
  subband stays registered with the input.  For real data the masks are cropped to
  the `rfft` half-plane.
"""
struct _DirectionalBank{A <: AbstractMatrix, V <: AbstractVector}
    ana::Vector{A}
    syn::Vector{A}
    phase_r::Vector{V}                  # e^{jω₁dₖ₁} along the rows (decimated only)
    phase_c::Vector{V}                  # e^{jω₀dₖ₀} along the columns
    delay::Vector{NTuple{2, Int}}       # (rows, cols) delay dₖ (decimated only), on the host
    sampling::Vector{NTuple{2, Int}}    # (rows, cols) decimation; (0, 0) quincunx; (1, 1) none
    dims::NTuple{2, Int}
    levels::Int
end

_is_decimated(b::_DirectionalBank) = !isempty(b.phase_r)

function _check_half(half::Symbol)
    half in (:both, :first, :second) ||
        throw(ArgumentError("half must be :both, :first or :second (got :$(half))"))
    return half
end

# Number of subbands of an `l`-level bank, full or half tree.
_bank_n_subbands(l::Int, half::Symbol) = l == 0 ? 1 : (half === :both ? 2^l : 2^(l - 1) + 1)

function _check_bank_dims(dims::NTuple{2, Int}, l::Int, decimated::Bool)
    decimated || return dims
    m = 2^max(l - 1, 1)
    all(n -> n % m == 0, dims) || throw(
        ArgumentError(
            "an $(l)-level DFB needs both image dimensions divisible by $(m) (got $(dims[1]) × $(dims[2]))"
        )
    )
    return dims
end

"""
    _directional_bank(Tf, dims, l, f_ladder; decimated, half=:both, real_data=false, upsample=0)

Build the equivalent filters of an `l`-level DFB (`decimated = true`) or NSDFB on a
`dims` grid from the ladder filter `f_ladder`, in precision `Tf`, as host arrays.
`real_data` crops nonsubsampled masks to the `rfft` half-plane.  `upsample = m`
replaces every nonsubsampled filter Hₖ(z) by Hₖ(z^(2^m I)) (da Cunha et al. 2006,
§II-A.3), which keeps the perfect-reconstruction identity Σₖ GₖHₖ = 1.
"""
function _directional_bank(
        ::Type{Tf}, dims::NTuple{2, Int}, l::Int, f_ladder::AbstractVector;
        decimated::Bool, half::Symbol = :both, real_data::Bool = false, upsample::Int = 0
    ) where {Tf <: AbstractFloat}
    _check_half(half)
    upsample >= 0 || throw(ArgumentError("upsample must be ≥ 0 (got $(upsample))"))
    decimated && upsample > 0 &&
        throw(ArgumentError("only the nonsubsampled bank takes upsampled filters"))
    _check_bank_dims(dims, l, decimated)
    n1, n2 = dims
    chs = _bank_channels(n1, n2, l, f_ladder; half = half)
    if decimated
        samp = map(_channel_sampling, chs)
        fold(s) = s == (0, 0) ? 1.0 : 1.0 / (s[1] * s[2])
        ana = [Tf.(fold(s) .* c.ana) for (c, s) in zip(chs, samp)]
        syn = [Tf.(c.syn) for c in chs]
        pr = [Complex{Tf}[cis(2π * c.delay[2] * k / n1) for k in 0:(n1 - 1)] for c in chs]
        pc = [Complex{Tf}[cis(2π * c.delay[1] * k / n2) for k in 0:(n2 - 1)] for c in chs]
        delay = [(c.delay[2], c.delay[1]) for c in chs]
    else
        # Hₖ(z^(2^m I)) has the frequency response Hₖ(2^m ω): sample the mask there.
        rows = mod.((2^upsample) .* (0:(real_data ? n1 ÷ 2 : n1 - 1)), n1) .+ 1
        cols = mod.((2^upsample) .* (0:(n2 - 1)), n2) .+ 1
        scale(c) = 2.0^(-length(c.path) / 2)
        ana = [Tf.(scale(c) .* c.ana[rows, cols]) for c in chs]
        syn = [Tf.(scale(c) .* c.syn[rows, cols]) for c in chs]
        pr = Vector{Complex{Tf}}[]
        pc = Vector{Complex{Tf}}[]
        delay = NTuple{2, Int}[]
        samp = fill((1, 1), length(chs))
    end
    return _DirectionalBank(ana, syn, pr, pc, delay, samp, dims, l)
end

# ── Quincunx storage of depth-1 channels ──────────────────────────────────────
#
# A depth-1 channel keeps the samples of the quincunx lattice {n₀ + n₁ even}.  They
# are stored column by column: column c of the n₁/2 × n₂ subband holds rows
# 0, 2, 4, … of an even column and rows −1, 1, 3, … of an odd one (0-based), i.e.
# y[r, c] = v[2r − (c mod 2), c].

function _quincunx_pack!(y::AbstractMatrix, V::AbstractMatrix)
    n1 = size(V, 1)
    m = n1 ÷ 2
    _store!(view(y, :, 1:2:size(y, 2)), view(V, 1:2:n1, 1:2:size(V, 2)))
    _store!(view(y, 2:m, 2:2:size(y, 2)), view(V, 2:2:(n1 - 2), 2:2:size(V, 2)))
    _store!(view(y, 1:1, 2:2:size(y, 2)), view(V, n1:n1, 2:2:size(V, 2)))
    return y
end

function _quincunx_unpack!(V::AbstractMatrix, y::AbstractMatrix)
    n1 = size(V, 1)
    m = n1 ÷ 2
    fill!(V, zero(eltype(V)))
    view(V, 1:2:n1, 1:2:size(V, 2)) .= view(y, :, 1:2:size(y, 2))
    view(V, 2:2:(n1 - 2), 2:2:size(V, 2)) .= view(y, 2:m, 2:2:size(y, 2))
    view(V, n1:n1, 2:2:size(V, 2)) .= view(y, 1:1, 2:2:size(y, 2))
    return V
end

# Write a complex intermediate into a subband of the data type.
_store!(y::AbstractArray{<:Real}, Y::AbstractArray) = (y .= real.(Y); y)
_store!(y::AbstractArray, Y::AbstractArray) = (y .= Y; y)
