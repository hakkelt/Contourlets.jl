# CDF 9/7 biorthogonal filter pair (Cohen-Daubechies-Feauveau).
#
# These are the standard JPEG2000 Part-1 (ISO/IEC 15444-1) low-pass filter
# coefficients.  The analysis filter h has 9 taps (symmetric, centred at index 5)
# and the synthesis filter g has 7 taps (symmetric, centred at index 4).
#
# Key properties:
#   • sum(h) ≈ 1  (unit DC gain)
#   • sum(g) ≈ 2  (compensates the factor-2 upsampling in the LP synthesis step)
#   • Together they form a biorthogonal pair with near-orthogonal response
#
# Reference: Cohen, Daubechies & Feauveau (1992), "Biorthogonal bases of compactly
# supported wavelets", Communications on Pure and Applied Mathematics 45(5).

"""
    CDF97 :: FilterPair{Float64}

The standard CDF 9/7 biorthogonal filter pair.  The 9-tap analysis filter and 7-tap
synthesis filter are the JPEG2000 default low-pass filters.

```jldoctest
julia> using Contourlets
julia> isapprox(sum(CDF97.h), 1.0; atol=1e-14)
true
julia> isapprox(sum(CDF97.g), 2.0; atol=1e-14)
true
```
"""
const CDF97 = FilterPair{Float64}(
    # analysis low-pass h (9 taps, symmetric, centre tap index 5)
    Float64[
        0.02674875741081,
        -0.016864118442875,
        -0.07822326652899,
        0.266864118442875,
        0.60294901823636,
        0.266864118442875,
        -0.07822326652899,
        -0.016864118442875,
        0.02674875741081,
    ],
    # synthesis low-pass g (7 taps, symmetric, centre tap index 4)
    Float64[
        -0.09127176311425,
        -0.0575435262285,
        0.59127176311425,
        1.115087052456994,
        0.59127176311425,
        -0.0575435262285,
        -0.09127176311425,
    ],
)
