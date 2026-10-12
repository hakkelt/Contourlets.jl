# Translation-invariant (TICT) and semi-translation-invariant (STICT) contourlet
# transforms of Eslami & Radha (2006, §III).
#
# Both start from the translation-invariant Laplacian pyramid (`tilp.jl`): at
# every level i, four full-size highpass channels K_c and, after the last level, a
# full-size lowpass.  A directional filter bank then splits every highpass
# channel (§III-C):
#
#   • TICT — the translation-invariant DFB, i.e. the DFB with its samplers
#     suppressed and the 2^(-l) scaling (§III-B).  That is this package's NSDFB.
#   • STICT — the critically sampled DFB.  The pyramid stays translation
#     invariant, the directional stage does not.  Redundancy 4L + 1.
#
# Following the channels' passbands (their Fig. 6), channels K₀ and K₃ get the full
# DFB and K₁ and K₂ the "vertical" and "horizontal" DFBs, half trees that stop
# splitting one of the two level-1 fans (Fig. 7(b)–(c)).  Which fan to split is
# fixed by the channel's energy: K₁ (coset shift (1,0) along the rows) carries most
# of its energy in the fan |ω_row| > |ω_col| and K₂ in the other, so each splits
# the fan its energy is in.  A level of depth lᵢ ≥ 1 then has 3·2^(lᵢ) + 2 TICT
# subbands (4 when lᵢ = 0), plus the lowpass.
#
# The DFB filters are the same at every pyramid level, as in the paper.  (The
# NSCT's coarse-scale upsampling, `nsdfb.jl`, does not help here: the coarse TILP
# channels carry the decimated pyramid's aliasing at high frequencies, and
# measured the same way the unmodified filters leak least at every level.)
#
# Everything is a pointwise product on the DFT grid, so both transforms are exact
# on the torus and the TICT commutes with every circular shift of the input.
#
# Reference: R. Eslami & H. Radha, "Translation-invariant contourlet transform and
# its application to image denoising", IEEE TIP 15(11), 2006, pp. 3362–3374.
#
# This file holds what the two share: the half tree of each channel, the
# input / element-type checks, the coefficient containers and the
# workspace helpers of the allocating forms.  The TICT is in `tict.jl`, the STICT
# in `stict.jl`.

# Half tree of each TILP channel (see above).
const _TILP_HALVES = (:both, :second, :first, :both)

_tict_types(params::ContourletParams{Tp}, T::Type) where {Tp} = (Td = promote_type(Tp, T); (Td, _filter_eltype(Td)))

# ── Coefficient containers ────────────────────────────────────────────────────

_tict_n_subbands(l::Int, c::Int) = _bank_n_subbands(l, _TILP_HALVES[c])

"""
    similar_tict_coefficients(params, image_size; Td=Float64, M=Matrix{Td}) -> TICTCoefficients

Zero-filled TICT coefficients for `image_size`: every subband has the image's size.
"""
function similar_tict_coefficients(
        params::ContourletParams{Tf}, image_size::Tuple{Int, Int};
        Td::Type = Tf, M::Type{<:AbstractMatrix} = Matrix{Td}
    ) where {Tf}
    z() = _allocate_zeros(M, Td, image_size)
    subbands = [[[z() for _ in 1:_tict_n_subbands(l, c)] for c in 1:4] for l in params.L_array]
    return TICTCoefficients(z(), subbands)
end

"""
    similar_stict_coefficients(params, image_size; Td=Float64, M=Matrix{Td}) -> TICTCoefficients

Zero-filled STICT coefficients for `image_size`: each channel's subbands have the
sizes of a critically sampled DFB of that channel.
"""
function similar_stict_coefficients(
        params::ContourletParams{Tf}, image_size::Tuple{Int, Int};
        Td::Type = Tf, M::Type{<:AbstractMatrix} = Matrix{Td}
    ) where {Tf}
    z(dims) = _allocate_zeros(M, Td, dims)
    n1, n2 = image_size
    subbands = [
        [[z(s) for s in dfb_subband_sizes(n1, n2, l; half = h)] for h in _TILP_HALVES]
            for l in params.L_array
    ]
    return TICTCoefficients(z(image_size), subbands)
end

# ── Checks and workspace helpers ──────────────────────────────────────────────

function _check_tict_input(ws, x::AbstractMatrix)
    size(x) == ws.image_size || throw(
        DimensionMismatch("image size $(size(x)) does not match the workspace's $(ws.image_size)")
    )
    return x
end

# The `!` forms write through the workspace's FFT plans, so the caller's arrays
# must have the workspace's element type.
function _check_tict_eltype(ws, A::AbstractMatrix, what::String)
    eltype(A) === eltype(ws) || throw(
        ArgumentError(
            string(
                "the $(what) has element type $(eltype(A)) but the workspace was built ",
                "for $(eltype(ws)); build both with the same element type"
            )
        )
    )
    return A
end

# Workspace at the data's own precision: like `nsct_forward`, the allocating forms
# never promote the data to the precision of `params`.
function _own_workspace(make, x::AbstractMatrix, params::ContourletParams)
    Td = _data_eltype(x)
    p = _convert_params(_filter_eltype(Td), params)
    return make(p, size(x); T = Td, M = typeof(similar(x, Td, 0, 0)))
end

# Input in the workspace's data type (a copy only when the types differ).
_ws_input(ws, x::AbstractMatrix) = (Td = eltype(ws); Td === eltype(x) ? x : Td.(x))
