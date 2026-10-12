# Quincunx filter pair for the Directional Filter Bank: the "23-45" biorthogonal
# filters of Phoong, Kim, Vaidyanathan & Ansari (1995).
#
# Phoong et al. parameterise a two-channel biorthogonal filter bank by a single
# 1-D filter β (their ladder structure, Fig. 2) and map it to the quincunx lattice
# by replacing β(z) with β(z₀z₁⁻¹)β(z₀z₁) (§V, eq. 32).  With the 12-tap β of their
# Example 3.2 the 2-D analysis filters have 23 × 23 and 45 × 45 support, hence the
# name; these are the filters Do & Vetterli (2005) use in the contourlet transform.
# The directional filter banks evaluate that 2-D pair in closed form (see
# `directional/bank_design.jl`); the ladder filter `f_ladder` is all they need.
#
# The shear-based two-channel bank `qfb_decompose` / `qfb_reconstruct` runs the
# 1-D ladder directly:
#
#   analysis:   y0 = (1/√2)·(p0 − B₁(p1))        p0/p1 = even/odd polyphase
#               y1 = −√2·p1 − B₀(y0)
#   synthesis:  p1 = (−1/√2)·(y1 + B₀(y0))
#               p0 = √2·y0 + B₁(p1)
#
# where B₀/B₁ filter with β modulated by π (odd taps negated, which turns the
# diamond-shaped pair into the fan-shaped one), and B₁ carries an extra one-sample
# shift aligning the two phases.  The synthesis ladder inverts the analysis ladder
# step by step, so perfect reconstruction holds for any β.
#
# References:
#   Phoong, Kim, Vaidyanathan & Ansari (1995), "A new class of two-channel
#     biorthogonal filter banks and wavelet bases", IEEE Trans. SP 43(3).
#   Do & Vetterli (2005), "The contourlet transform", IEEE Trans. IP 14(12).

# The 12-tap linear-phase β of Phoong et al. (1995), Example 3.2 (N = 6, designed
# with the McClellan–Parks algorithm): v₁ … v₆, the half of the symmetric impulse
# response β(z) = Σₖ vₖ (z^-(N-k) + z^-(N+k-1)).
const _PKVA12_HALF = [0.63, -0.193, 0.0972, -0.0526, 0.0272, -0.0144]

"""
    pkva_ladder_filter() -> Vector{Float64}

The symmetric 12-tap ladder filter β of Phoong et al. (1995, Example 3.2) used by
the default DFB filter pair [`Q2345`](@ref).
"""
pkva_ladder_filter() = [reverse(_PKVA12_HALF); _PKVA12_HALF]

# Modulate the ladder filter by π (negate every odd, 1-based, tap): β(z) → β(−z).
# In the 1-D ladder of the shear-based QFB this turns the diamond pair into the fan
# pair.
function _ladder_modulate(f::AbstractVector{T}) where {T}
    return T[isodd(k) ? -f[k] : f[k] for k in eachindex(f)]
end

# `Q2345.f_ladder` stores the *unmodulated* symmetric β; the shear-based QFB
# applies the fan modulation once at its use site via `_ladder_modulate`.

"""
    Q2345 :: QuincunxFilterPair{Float64}

The "23-45" biorthogonal quincunx filter pair of Phoong et al. (1995) for the DFB
stage, parameterised by the 12-tap ladder filter β of their Example 3.2 (stored in
`f_ladder`).  The directional filter banks use
its 2-D quincunx mapping, whose analysis filters have 23 × 23 and 45 × 45 support.
Perfect reconstruction is structural — exact to machine precision at every tree
depth.

Perfect-reconstruction property:

```jldoctest
julia> using Contourlets

julia> check_pr_condition(Q2345)
true
```
"""
const Q2345 = QuincunxFilterPair{Float64}(pkva_ladder_filter())
