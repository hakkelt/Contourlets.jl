# Contourlet Transform Example

This example applies the Discrete Contourlet Transform to a random image and
verifies perfect reconstruction.

```@example ct
using Contourlets, TestImages, ImageCore, Colors, Plots, Statistics
Plots.default(show = false)

# Load a natural image (512×512)
img = Float64.(Gray.(testimage("barbara")))[1:512, 100:611]

# Build params: J=3 scales with parabolic direction counts 2/4/4
params = ContourletParams(J=3, L_array=parabolic_levels(3))
println("Direction counts: ", params.L_array)  # [1, 2, 2]
```

---

## Variant 1 — Allocating (simplest)

`ct_forward` and `ct_inverse` allocate all output buffers internally.  Use this
for one-shot processing where allocation cost is irrelevant.

```@example ct
coeffs = ct_forward(img, params)

println("Coarse size:        ", size(coeffs.coarse))
println("Subbands per scale: ", map(length, coeffs.subbands))

rec = ct_inverse(coeffs, params)
err = maximum(abs, rec .- img)
println("PR error: ", err)   # < 1e-12
@assert err < 1e-12
```

---

## Variant 2 — In-place (preallocated output)

`ct_forward!` and `ct_inverse!` write into a caller-supplied output container,
eliminating the top-level allocation while still allocating scratch buffers
internally.  Use this when you want to control the lifetime of the coefficient
and image buffers, e.g. when reusing them across outer iterations.

```@example ct
# Allocate output containers once
coeffs_buf = similar_coefficients(params, size(img))
img_out    = similar(img)

ct_forward!(coeffs_buf, img, params)          # writes into coeffs_buf
ct_inverse!(img_out, coeffs_buf, params)      # writes into img_out

err = maximum(abs, img_out .- img)
@assert err < 1e-12
println("In-place PR error: ", err)
```

---

## Variant 3 — Allocating + workspace

Pass a preallocated [`ContourletWorkspace`](@ref) via `workspace` to reuse
internal pyramid scratch buffers across calls.  The output coefficients and
image are still freshly allocated on each call; only the pyramid stage is
allocation-free.

```@example ct
ws     = make_workspace(params, size(img))

coeffs = ct_forward(img, params; workspace=ws)   # pyramid scratch reused
rec    = ct_inverse(coeffs, params; workspace=ws)

err = maximum(abs, rec .- img)
@assert err < 1e-12
println("Workspace (allocating) PR error: ", err)
```

---

## Variant 4 — In-place + workspace (zero-allocation per call)

Combining the in-place forms with a workspace makes the hot path entirely
allocation-free: the pyramid scratch buffers and the coefficient/image
containers are all preallocated.  This is the recommended pattern for
tight iterative algorithms.

```@example ct
ws         = make_workspace(params, size(img))
coeffs_buf = similar_coefficients(params, size(img))
img_out    = similar(img)

# Warm-up call (workspace is already prewarmed by make_workspace; this
# ensures the specific argument types are JIT-compiled before timing)
ct_forward!(coeffs_buf, img, params; workspace=ws)
ct_inverse!(img_out, coeffs_buf, params; workspace=ws)

# Iterative algorithm — no per-iteration allocation
for iter in 1:100
    ct_forward!(coeffs_buf, img, params; workspace=ws)

    # ... modify coeffs_buf.subbands here (e.g. soft-threshold) ...

    ct_inverse!(img_out, coeffs_buf, params; workspace=ws)
end

err = maximum(abs, img_out .- img)
println("Zero-alloc iterative PR error: ", err)  # < 1e-12
```

---

## Understanding the Coefficients

`ct_forward` returns a [`ContourletCoefficients`](@ref) struct:

| Field | Type | Description |
|-------|------|-------------|
| `coarse` | `Matrix{T}` | Low-frequency residual (size ≈ N/2^J × M/2^J) |
| `subbands[j]` | `Vector{Matrix{T}}` | `2^L_array[j]` directional subbands at scale `j` |

The subbands are ordered by direction angle: subband `k` covers the
frequency wedge centred at angle ``\pi(k-1)/2^{L_j}``.

Use [`similar_coefficients`](@ref) to preallocate a matching buffer for
in-place use, and [`estimate_workspace_size`](@ref) to inspect the scratch
footprint before constructing the workspace.

### Visualizing the coefficients

We can render the coarse residual together with every directional subband of the
*Barbara* image.  Each directional subband is critically sampled, so coefficients
at finer scales live on smaller grids; notice how each subband responds most
strongly to edges aligned with its frequency wedge.  Each subband is shown with
its own symmetric grey scale (mid-grey = 0, clipped at the 99th percentile of
`|coefficient|`) so its directional structure stays visible despite the wide
range of coefficient magnitudes across scales.

```@example ct
function plot_coefficients(coeffs)
    panels = Any[]
    push!(panels, heatmap(coeffs.coarse; title = "coarse", color = :grays,
        aspect_ratio = :equal, axis = false, colorbar = false, yflip = true))
    for j in eachindex(coeffs.subbands), k in eachindex(coeffs.subbands[j])
        sb = coeffs.subbands[j][k]
        c  = max(quantile(abs.(vec(sb)), 0.99), eps())   # per-subband contrast
        push!(panels, heatmap(sb; title = "scale $j dir $k", color = :grays,
            clims = (-c, c), aspect_ratio = :equal, axis = false,
            colorbar = false, yflip = true))
    end
    n = length(panels)
    cols = ceil(Int, sqrt(n))
    rows = ceil(Int, n / cols)
    plot(panels...; layout = (rows, cols), size = (240 * cols, 240 * rows))
end

plot_coefficients(ct_forward(img, params))
savefig("ct_coeffs.png"); nothing # hide
```

![](ct_coeffs.png)
