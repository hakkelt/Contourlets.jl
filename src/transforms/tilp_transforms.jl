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
# Levels are numbered as the coefficients are stored: level 1 is the coarsest,
# level L the finest (the cascade's first split).
#
# Everything is a pointwise product on the DFT grid, so both transforms are exact
# on the torus and the TICT commutes with every circular shift of the input.
#
# Reference: R. Eslami & H. Radha, "Translation-invariant contourlet transform and
# its application to image denoising", IEEE TIP 15(11), 2006, pp. 3362–3374.
#
# This file holds what the two share: the half tree of each channel and the
# subband counts and offsets.  The TICT is in `tict.jl`, the STICT in `stict.jl`.

# Half tree of each TILP channel (see above).
const _TILP_HALVES = (:both, :second, :first, :both)

_tict_n_subbands(l::Int, c::Int) = _bank_n_subbands(l, _TILP_HALVES[c])

# Subband (i, c, k) is slice `offsets[4(i - 1) + c] + k` of TICT coefficients.
function _tilp_offsets(t::Union{TICT, STICT})
    offsets = Vector{Int}(undef, 4 * t.scales)
    s = 1
    for (i, l) in enumerate(t.direction_levels), c in 1:4
        offsets[4 * (i - 1) + c] = s
        s += _tict_n_subbands(l, c)
    end
    return offsets
end
