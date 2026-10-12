# [2-D transforms](@id guide2d)

Contourlets.jl offers four 2-D transforms over the same building blocks — a
multiscale pyramid followed by a directional filter bank:

| | [`ct_forward`](@ref) (CT) | [`nsct_forward`](@ref) (NSCT) | [`tict_forward`](@ref) (TICT) | [`stict_forward`](@ref) (STICT) |
|---|---|---|---|---|
| Pyramid | Laplacian | nonsubsampled | translation-invariant Laplacian | translation-invariant Laplacian |
| Directional stage | decimated | nonsubsampled | nonsubsampled | decimated |
| Shift invariance | no | exact | exact | pyramid only |
| Redundancy | ≈ 4/3 | `1 + Σ_j 2^{L_j}` | `1 + 2J + 3 Σ_j 2^{L_j}` | `1 + 4J` |
| Typical use | compression, M-term approximation | denoising, feature extraction | denoising | denoising at low redundancy |

The CT and NSCT return a [`ContourletCoefficients`](@ref) / [`NSCTCoefficients`](@ref):
a `coarse` low-pass residual plus `2^L_array[j]` directional subbands at every
scale `j`, ordered by angle.  The TICT and STICT, described [below](@ref guide2d-tict),
return a [`TICTCoefficients`](@ref) with four channels per scale.  For
volumes, see [Surfacelets.jl](https://github.com/hakkelt/Surfacelets.jl); for the underlying filter banks,
[Theory](@ref theory).

```@setup guide2d
using Contourlets, TestImages, ImageCore, Colors, Plots, Statistics
Plots.default(show = false)
```

## The test image

```@example guide2d
using Contourlets, TestImages, ImageCore, Colors, Plots, Statistics
Plots.default(show = false)

img    = Float64.(Gray.(testimage("barbara")))[1:512, 100:611]
params = ContourletParams(J = 3, L_array = parabolic_levels(3))

println("direction exponents: ", params.L_array)   # 2^L subbands per scale
```

[`parabolic_levels`](@ref) applies the parabolic scaling rule (width ≈ length²),
doubling the direction count every other scale.

## Contourlet transform

```@example guide2d
coeffs = ct_forward(img, params)
rec    = ct_inverse(coeffs, params)

println("coarse size:        ", size(coeffs.coarse))
println("subbands per scale: ", map(length, coeffs.subbands))
println("PR error:           ", maximum(abs, rec .- img))
@assert maximum(abs, rec .- img) < 1e-12
```

Perfect reconstruction is *structural*: the quincunx filters of Phoong et al.
(1995) satisfy the perfect-reconstruction conditions for any ladder filter, so
the residual sits at the round-off floor at every directional depth.

Every subband is critically sampled, so finer scales live on smaller grids. Each
panel below is drawn on its own symmetric grey scale (mid-grey = 0, clipped at
the 99th percentile of `|coefficient|`) — without that, the magnitude spread
across scales washes the fine subbands out.

```@example guide2d
function plot_coefficients(coeffs; clip = 0.99)
    panels = Any[heatmap(coeffs.coarse; title = "coarse", color = :grays,
        aspect_ratio = :equal, axis = false, colorbar = false, yflip = true)]
    for j in eachindex(coeffs.subbands), k in eachindex(coeffs.subbands[j])
        sb = coeffs.subbands[j][k]
        c  = max(quantile(abs.(vec(sb)), clip), eps())
        push!(panels, heatmap(sb; title = "scale $j dir $k", color = :grays,
            clims = (-c, c), aspect_ratio = :equal, axis = false,
            colorbar = false, yflip = true))
    end
    cols = ceil(Int, sqrt(length(panels)))
    rows = ceil(Int, length(panels) / cols)
    return plot(panels...; layout = (rows, cols), size = (240 * cols, 240 * rows))
end

plot_coefficients(coeffs)
savefig("ct_coeffs.png"); nothing # hide
```

![](ct_coeffs.png)

Each subband responds most strongly to edges aligned with its frequency wedge —
the fabric patterns of *Barbara* separate by orientation.

## Nonsubsampled contourlet transform

The NSCT drops every downsampler, so subbands and coarse residual all keep the
input resolution and a circular shift of the input shifts every subband by the
same amount.

```@example guide2d
ns_params = ContourletParams(J = 2, L_array = [2, 3])

ns  = nsct_forward(img, ns_params)
rec = nsct_inverse(ns, ns_params)

println("coarse size: ", size(ns.coarse), " (input ", size(img), ")")
println("PR error:    ", maximum(abs, rec .- img))
@assert maximum(abs, rec .- img) < 2e-15
```

Shift invariance is a property the test suite enforces, not an approximation:

```@example guide2d
shift      = (3, 7)
ns_shifted = nsct_forward(circshift(img, shift), ns_params)

err = maximum(
    maximum(abs, ns_shifted.subbands[j][k] .- circshift(ns.subbands[j][k], shift))
        for j in 1:ns_params.J for k in eachindex(ns.subbands[j])
)
println("shift equivariance error: ", err)
@assert err < 1e-10
```

NSCT subbands are sparse — mostly near-zero with occasional strong edge
responses — so they are rendered with a lower clip (75th percentile) than the
critically sampled CT above:

```@example guide2d
plot_coefficients(ns; clip = 0.75)
savefig("nsct_coeffs.png"); nothing # hide
```

![](nsct_coeffs.png)

## [Translation-invariant contourlets: TICT and STICT](@id guide2d-tict)

Eslami & Radha (2006) make the contourlet transform translation invariant the
other way round from the NSCT: they keep the Laplacian pyramid's filters and take
*all shifts* of it, which gives the translation-invariant Laplacian pyramid
(TILP).  Each TILP level has four full-size highpass channels — the four
polyphase components of the pyramid's detail image, for every position of the
sampling lattice — instead of the NSCT's one.  A directional filter bank then
splits every channel:

- [`tict_forward`](@ref) uses the nonsubsampled DFB: the whole transform is
  translation invariant.
- [`stict_forward`](@ref) uses the critically sampled DFB: only the pyramid is
  translation invariant, and the redundancy drops to `4J + 1`.

Channels 1 and 4 get the full directional tree; channels 2 and 3 pass mostly row
and mostly column frequencies, so they get half trees that split only that fan
(see [Theory](@ref theory-tict)).

```@example guide2d
ti_params = ContourletParams(J = 2, L_array = [3, 2])

ti  = tict_forward(img, ti_params)
sti = stict_forward(img, ti_params)

subbands(c) = sum(length, c.subbands[j][ch] for j in 1:ti_params.J for ch in 1:4)
redundancy(c) = (sum(length, Iterators.flatten(Iterators.flatten(c.subbands))) +
                 length(c.coarse)) / length(img)

println("TICT:  ", subbands(ti), " subbands, redundancy ", redundancy(ti))
println("STICT: ", subbands(sti), " subbands, redundancy ", redundancy(sti))
println("PR errors: ", maximum(abs, tict_inverse(ti, ti_params) .- img), ", ",
    maximum(abs, stict_inverse(sti, ti_params) .- img))
```

The coefficients are indexed `c.subbands[scale][channel][direction]`.  Like the
NSCT, the TICT commutes with circular shifts exactly:

```@example guide2d
ti_shifted = tict_forward(circshift(img, shift), ti_params)
err = maximum(
    maximum(abs, ti_shifted.subbands[j][ch][d] .- circshift(ti.subbands[j][ch][d], shift))
        for j in 1:ti_params.J for ch in 1:4 for d in eachindex(ti.subbands[j][ch])
)
println("shift equivariance error: ", err)
@assert err < 1e-10
```

The second level of each, one row per channel, shows the difference.  TICT
subbands keep the image's size; STICT subbands are those of a critically sampled
DFB of each full-size channel.  Full-size subbands alias into moiré at this
panel size, so each panel shows the same central window — a quarter of the
subband's height and width:

```@example guide2d
function crop_center(sb, frac = 4)
    r = size(sb, 1) ÷ 2 .+ (-(size(sb, 1) ÷ 2frac):(size(sb, 1) ÷ 2frac))
    c = size(sb, 2) ÷ 2 .+ (-(size(sb, 2) ÷ 2frac):(size(sb, 2) ÷ 2frac))
    return sb[r, c]
end

function plot_tict_level(c, j; clip = 0.99)
    nd = maximum(length, c.subbands[j])
    panels = Any[]
    for ch in 1:4, d in 1:nd
        if d <= length(c.subbands[j][ch])
            sb = crop_center(c.subbands[j][ch][d])
            cl = max(quantile(abs.(vec(sb)), clip), eps())
            push!(panels, heatmap(sb; title = "channel $ch dir $d", color = :grays,
                clims = (-cl, cl), aspect_ratio = :equal, axis = false,
                colorbar = false, yflip = true))
        else
            push!(panels, plot(; framestyle = :none, grid = false, ticks = false))
        end
    end
    return plot(panels...; layout = (4, nd), size = (200 * nd, 800))
end

plot_tict_level(ti, 2)
savefig("tict_coeffs.png"); nothing # hide
```

![](tict_coeffs.png)

```@example guide2d
plot_tict_level(sti, 2)
savefig("stict_coeffs.png"); nothing # hide
```

![](stict_coeffs.png)

Both transforms are FFT-based and have workspaces —
[`make_tict_workspace`](@ref), [`make_stict_workspace`](@ref) — with which the
`!` forms allocate nothing.

## Choosing an API variant

Every transform comes in four forms. They compute the same thing; they differ
only in who owns the output container and the scratch buffers.

| Form | Output container | Scratch buffers | Use when |
|---|---|---|---|
| `ct_forward(img, params)` | allocated | allocated | one-shot processing |
| `ct_forward!(dst, img, params)` | caller | allocated | you control the output's lifetime |
| `ct_forward(img, params; workspace = ws)` | allocated | reused | repeated calls, output kept per call |
| `ct_forward!(dst, img, params; workspace = ws)` | caller | reused | iterative algorithms — no per-call allocation |

The `!` and non-`!` entry points take identical arguments; the only difference
is the leading destination. [`make_workspace`](@ref) (CT) and
[`make_nsct_workspace`](@ref) (NSCT) build the scratch,
[`similar_coefficients`](@ref) / [`similar_nsct_coefficients`](@ref) the
coefficient container, and `Base.summarysize(ws)` reports a workspace's
actual memory footprint.

```@example guide2d
ws         = make_nsct_workspace(ns_params, size(img))   # prewarm = true by default
coeffs_buf = similar_nsct_coefficients(ns_params, size(img))
img_out    = similar(img)

for _ in 1:10
    nsct_forward!(coeffs_buf, img, ns_params; workspace = ws)

    # … threshold / prox step on coeffs_buf.subbands …

    nsct_inverse!(img_out, coeffs_buf, ns_params; workspace = ws)
end

println("PR error after 10 iterations: ", maximum(abs, img_out .- img))
```

The NSCT workspace is where this matters most: it caches per-scale FFT plans and
precomputed filter spectra, which turns each call into an FFT, a pointwise
multiply and an inverse FFT. See [Performance](@ref performance) for what that
buys.

!!! note "Threading"
    A host NSCT or TICT workspace keeps one FFT buffer per Julia thread, and with
    threading on (`threading = Enabled()`, or the default for complex data) the
    `!` transforms split their independent subband FFTs between the threads.  The
    choice is made per call.

## Using the stages on their own

The pyramid and the directional filter bank are public, so a single stage can be
used without the full transform:

```@example guide2d
coarse, bandpass = lp_decompose(img, CDF97)       # one Laplacian pyramid level
back             = lp_reconstruct(coarse, bandpass, CDF97)

p = plot(layout = (1, 2), size = (800, 400))
heatmap!(p[1], coarse, title = "coarse", color = :grays,
    aspect_ratio = :equal, axis = false, colorbar = false, yflip = true)
heatmap!(p[2], bandpass, title = "bandpass", color = :grays,
    aspect_ratio = :equal, axis = false, colorbar = false, yflip = true)
savefig(p, "lp_demo.png"); nothing # hide
```

![](lp_demo.png)

```@example guide2d
println("LP round-trip error: ", maximum(abs, back .- img))

sbs = dfb_decompose(bandpass, 3, Q2345)           # 2^3 directional subbands
println("DFB subbands: ", length(sbs), " of size ", size(sbs[1]))
println("DFB round-trip error: ",
    maximum(abs, dfb_reconstruct(sbs, Q2345) .- bandpass))
```

[`nsdfb_decompose`](@ref) / [`nsdfb_reconstruct`](@ref) are the nonsubsampled
counterparts. The complete list is in the [API Reference](@ref api).
