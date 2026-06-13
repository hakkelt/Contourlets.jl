# Contourlet Transform Example

This example applies the Discrete Contourlet Transform to a random image and
verifies perfect reconstruction.

```julia
using Contourlets, TestImages, ImageCore, Colors

# Load a natural image and crop to 128x128
img = Float64.(Gray.(testimage("barbara")))[100:227, 200:327]

# Build params with parabolic scaling: J=3 scales, 2/4/4 directions per scale
params = ContourletParams(J=3, L_array=parabolic_levels(3))
println("Direction counts: ", params.L_array)   # e.g. [1, 2, 2]

# ── Forward transform ───────────────────────────────────────────────────────
coeffs = ct_forward(img, params)

println("Coarse size:  ", size(coeffs.coarse))
println("Subbands per scale: ", map(length, coeffs.subbands))

# ── Perfect reconstruction ──────────────────────────────────────────────────
rec   = ct_inverse(coeffs)
err   = maximum(abs, rec .- img)
println("PR error: ", err)           # < 1e-12
@assert err < 1e-12

# ── Zero-allocation iterative usage ─────────────────────────────────────────
ws  = make_workspace(Float64, size(img), params)
buf = similar_coefficients(params, size(img))

for iter in 1:50
    ct_forward!(buf, img, ws)
    # ... modify buf.subbands here (e.g. soft-threshold) ...
    out = similar(img)
    ct_inverse!(out, buf, ws)
end
println("Iterative loop completed without error")
```

## Understanding the Coefficients

`ct_forward` returns a [`ContourletCoefficients`](@ref) struct:

| Field | Type | Description |
|-------|------|-------------|
| `coarse` | `Matrix{T}` | Low-frequency residual (size ≈ N/2^J × M/2^J) |
| `subbands[j]` | `Vector{Matrix{T}}` | `2^L_array[j]` directional subbands at scale `j` |
| `params` | `ContourletParams{T}` | Parameters used to produce these coefficients |

The subbands are ordered by direction angle: subband `k` covers the
frequency wedge centred at angle ``\pi(k-1)/2^{L_j}``.
