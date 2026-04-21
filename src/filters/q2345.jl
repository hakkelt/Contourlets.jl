# Quincunx filter pair for the Directional Filter Bank.
#
# Theory (Do & Vetterli 2005, Section V):
# After a shear operation the quincunx FB in the original domain reduces to a
# simple 1-D two-channel filter bank applied in the column direction, followed
# by column-stride-2 downsampling.  For the :h (horizontal) shear nodes, the
# 1-D filter operates along columns; for :v (vertical) shear nodes the same
# 1-D filter is applied along rows (equivalently, a transposed image is passed).
#
# PR conditions for a 2-channel 1-D FB with downsampling by 2:
#   H₀(z)G₀(z) + H₀(-z)G₀(-z) = 2           [distortion-free]
#   H₀(-z)G₀(z) + H₁(-z)G₁(z) = 0           [aliasing cancellation]
# where H₁(z) = H₀(-z) and G₁(z) = G₀(-z) (modulation by (-1)^k).
#
# The Haar pair below satisfies these conditions exactly:
#   H₀(z) = (1 + z⁻¹)/2                      analysis LP
#   H₁(z) = (1 − z⁻¹)/2                      analysis HP
#   G₀(z) = 1 + z                             synthesis LP
#   G₁(z) = 1 − z                             synthesis HP
#
# Filters are stored as 1-row matrices so that conv2d applies them
# in the column direction.  Row-direction application is handled in
# qfb_decompose/reconstruct by transposing the image.
#
# Reference: Vetterli & Kovačević (1995), "Wavelets and Subband Coding",
#            Chapters 4–5 (2-channel PR filter bank conditions).

"""
    Q2345 :: QuincunxFilterPair{Float64}

A two-channel PR quincunx filter pair for the DFB stage.  The stored
analysis kernel `h_q` (1×2) and synthesis kernel `g_q` (1×2) form a
Haar-type biorthogonal pair in the column direction.  The high-pass
filters are derived by modulation: `H₁(z) = H₀(-z)`, `G₁(z) = G₀(-z)`.

Perfect-reconstruction property:

```jldoctest
julia> using Contourlets
julia> check_pr_condition(Q2345)
true
```
"""
const Q2345 = let
    # H₀(z) = (1 + z⁻¹)/2  →  h_q[1,1] = 0.5  (lag 0), h_q[1,2] = 0.5  (lag +1)
    # stored as 1×2 matrix; origin (column 0-lag) at column index c_h[2] = 1
    h_q = Float64[0.5 0.5]

    # G₀(z) = 1 + z  →  g_q[1,1] = 1  (lag −1), g_q[1,2] = 1  (lag 0)
    # origin (column 0-lag) at column index c_g[2] = 2
    g_q = Float64[1.0 1.0]

    QuincunxFilterPair{Float64}(h_q, g_q, (1, 1), (1, 2))
end
