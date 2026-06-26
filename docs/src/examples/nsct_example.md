# Nonsubsampled Contourlet Transform Example

The NSCT is fully shift-invariant: a circular shift of the input produces the
same shift in every subband.  All subbands retain the full spatial resolution
of the input (no downsampling), so the transform is redundant but invariant.

```@example nsct
using Contourlets, TestImages, ImageCore, Colors, Plots, Statistics
Plots.default(show = false)

# Load a natural image (512×512)
img    = Float64.(Gray.(testimage("barbara")))[1:512, 100:611]
params = ContourletParams(J=2, L_array=[2, 3])
```

---

## Variant 1 — Allocating (simplest)

`nsct_forward` and `nsct_inverse` allocate all output buffers internally.

```@example nsct
ns  = nsct_forward(img, params)
rec = nsct_inverse(ns, params)

err = maximum(abs, rec .- img)
println("NSCT PR error: ", err)   # < 2e-15
@assert err < 2e-15

# All subbands and coarse have the same spatial size as the input
println("Coarse size: ", size(ns.coarse))
for j in 1:params.J
    println("Scale $j: $(length(ns.subbands[j])) subbands, each $(size(ns.subbands[j][1]))")
end
```

---

## Variant 2 — In-place (preallocated output)

`nsct_forward!` and `nsct_inverse!` write into caller-supplied containers.
Use this to control buffer lifetime when the allocation cost of the output
containers matters (e.g. in an outer loop where you want explicit control over
GC pressure).

```@example nsct
coeffs_buf = similar_nsct_coefficients(params, size(img))
img_out    = similar(img)

nsct_forward!(coeffs_buf, img, params)
nsct_inverse!(img_out, coeffs_buf, params)

err = maximum(abs, img_out .- img)
@assert err < 2e-15
println("In-place PR error: ", err)
```

---

## Variant 3 — Allocating + workspace

Pass a [`ContourletWorkspace`](@ref) from [`make_nsct_workspace`](@ref) via
`workspace` to reuse pyramid scratch buffers and FFT plans across calls.  The
NSCT workspace additionally caches per-scale FFT plans and precomputed filter
spectra; first-call latency is absorbed by workspace construction
(`prewarm=true` is the default).

```@example nsct
ws  = make_nsct_workspace(params, size(img))   # prewarm=true by default

ns  = nsct_forward(img, params; workspace=ws)
rec = nsct_inverse(ns, params; workspace=ws)

err = maximum(abs, rec .- img)
@assert err < 2e-15
println("Workspace (allocating) PR error: ", err)
```

!!! note "FFT threading"
    The FFT plans inside the workspace are bound to the thread count chosen at
    construction time (controlled by the `threading` kwarg of
    `make_nsct_workspace`).  Passing a conflicting `threading` kwarg at call
    time triggers a one-time warning; to change FFT threading, build a new
    workspace.

---

## Variant 4 — In-place + workspace (zero-allocation per call)

The combination of in-place forms with a prewarmed workspace is the
recommended pattern for iterative algorithms (e.g. proximal methods, iterative
denoising).  After the first call the hot path allocates nothing.

```@example nsct
ws         = make_nsct_workspace(params, size(img))   # prewarm=true
coeffs_buf = similar_nsct_coefficients(params, size(img))
img_out    = similar(img)

# Warm-up: ensures the specific types are JIT-compiled before timing
nsct_forward!(coeffs_buf, img, params; workspace=ws)
nsct_inverse!(img_out, coeffs_buf, params; workspace=ws)

# Iterative algorithm — no per-iteration allocation
for iter in 1:100
    nsct_forward!(coeffs_buf, img, params; workspace=ws)

    # ... threshold / modify coeffs_buf.subbands here ...

    nsct_inverse!(img_out, coeffs_buf, params; workspace=ws)
end

err = maximum(abs, img_out .- img)
println("Zero-alloc iterative PR error: ", err)   # < 2e-15
```

---

## Shift Invariance Demonstration

```@example nsct
shift       = (3, 7)
img_shifted = circshift(img, shift)

ns          = nsct_forward(img, params)
ns_shifted  = nsct_forward(img_shifted, params)

for j in 1:params.J, k in eachindex(ns.subbands[j])
    expected = circshift(ns.subbands[j][k], shift)
    actual   = ns_shifted.subbands[j][k]
    err_si   = maximum(abs, actual .- expected)
    @assert err_si < 1e-10
end
println("Shift invariance verified for all subbands")
```

---

## Understanding the Coefficients

`nsct_forward` returns a [`ContourletCoefficients`](@ref) struct with the same
layout as the CT (`coarse` plus `2^L_array[j]` directional subbands per scale),
but because the NSCT performs **no downsampling**, every subband — and the coarse
residual — keeps the full spatial size of the input.  The subbands are ordered by
direction angle: subband `k` covers the frequency wedge centred at angle
``\pi(k-1)/2^{L_j}``.

### Visualizing the coefficients

Rendering the coarse residual alongside every directional subband of the
*Barbara* image shows how each shift-invariant subband isolates edges aligned
with its frequency wedge — all at the full input resolution.  Each subband is
shown with its own symmetric grey scale (mid-grey = 0, clipped at a per-subband
quantile of `|coefficient|`) so its directional structure stays visible.  The
NSCT is redundant and its subbands are *sparse* — most coefficients are near
zero with occasional strong edge responses — so we clip at the 75th percentile
(rather than the 99th used for the critically-sampled CT) to keep the texture
from washing out to mid-grey.

```@example nsct
function plot_coefficients(coeffs)
    panels = Any[]
    push!(panels, heatmap(coeffs.coarse; title = "coarse", color = :grays,
        aspect_ratio = :equal, axis = false, colorbar = false, yflip = true))
    for j in eachindex(coeffs.subbands), k in eachindex(coeffs.subbands[j])
        sb = coeffs.subbands[j][k]
        c  = max(quantile(abs.(vec(sb)), 0.75), eps())   # per-subband contrast
        push!(panels, heatmap(sb; title = "scale $j dir $k", color = :grays,
            clims = (-c, c), aspect_ratio = :equal, axis = false,
            colorbar = false, yflip = true))
    end
    n = length(panels)
    cols = ceil(Int, sqrt(n))
    rows = ceil(Int, n / cols)
    plot(panels...; layout = (rows, cols), size = (240 * cols, 240 * rows))
end

plot_coefficients(nsct_forward(img, params))
savefig("nsct_coeffs.png"); nothing # hide
```

![](nsct_coeffs.png)

## When to Use NSCT vs CT

| Property | CT | NSCT |
|----------|----|----|
| Shift invariant | ✗ | ✓ |
| Redundancy | Low (≈ 4/3) | High (1 + Σ 2^L_j) |
| Complexity | O(N) | O(N log N) via FFT |
| Typical use | Compression | Feature extraction, denoising |
