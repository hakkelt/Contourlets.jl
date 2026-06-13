# Nonsubsampled Contourlet Transform Example

The NSCT is fully shift-invariant: a circular shift of the input produces the
same shift in every subband.

```julia
using Contourlets, TestImages, ImageCore, Colors

# Load a natural image and crop to 64x64 for the shift invariance demo
img = reverse(Float64.(Gray.(testimage("barbara")))[1:512, 100:611], dims=2)

params = ContourletParams(J=2, L_array=[2, 3])

# ── Forward / inverse ───────────────────────────────────────────────────────
ns = nsct_forward(img, params)
rec = nsct_inverse(ns)

err = maximum(abs, rec .- img)
println("NSCT PR error: ", err)      # < 2e-15
@assert err < 2e-15

# All subbands and coarse have the same spatial size as the input
println("Coarse size: ", size(ns.coarse))    # (64, 64)
for j in 1:params.J
    println("Scale $j: $(length(ns.subbands[j])) subbands, ",
            "each ", size(ns.subbands[j][1]))
end

# ── Shift invariance demonstration ──────────────────────────────────────────
shift = (3, 7)
img_shifted = circshift(img, shift)

ns_shifted = nsct_forward(img_shifted, params)

# Each subband of the shifted image equals the circshifted subband of the original
for j in 1:params.J, k in eachindex(ns.subbands[j])
    expected = circshift(ns.subbands[j][k], shift)
    actual   = ns_shifted.subbands[j][k]
    err_si   = maximum(abs, actual .- expected)
    println("Scale $j, dir $k shift error: ", err_si)
    @assert err_si < 1e-10
end
println("Shift invariance verified for all subbands")
```

## When to Use NSCT vs CT

| Property | CT | NSCT |
|----------|----|----|
| Shift invariant | ✗ | ✓ |
| Redundancy | Low (≈ 4/3) | High (1 + Σ 2^L_j) |
| Complexity | O(N) | O(N log N) via FFT |
| Typical use | Compression | Feature extraction, denoising |
