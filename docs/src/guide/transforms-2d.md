# [2-D transforms](@id guide2d)

Contourlets.jl offers four 2-D transforms over the same building blocks — a
multiscale pyramid followed by a directional filter bank:

| | [`CT`](@ref) | [`NSCT`](@ref) | [`TICT`](@ref) | [`STICT`](@ref) |
|---|---|---|---|---|
| Pyramid | Laplacian | nonsubsampled | translation-invariant Laplacian | translation-invariant Laplacian |
| Directional stage | decimated | nonsubsampled | nonsubsampled | decimated |
| Shift invariance | no | exact | exact | pyramid only |
| Redundancy | ≈ 4/3 | `1 + Σ_j 2^{L_j}` | `1 + 2J + 3 Σ_j 2^{L_j}` | `1 + 4J` |
| Typical use | compression, M-term approximation | denoising, feature extraction | denoising | denoising at low redundancy |

Each transform is an immutable value built from keywords, such as
`CT(J = 3, L_array = [2, 2, 1])`, and is run with [`analyze`](@ref) and
[`synthesize`](@ref).  The NSCT and TICT, whose subbands all have the image's
size, return one array of size `(size(img)..., n)` with subband `i` in
`c[:, :, i]`.  The CT and STICT, whose subbands have different sizes, return a
[`SubbandVector`](@ref): a vector over one flat storage vector (`parent(c)`)
that broadcasts and reduces like any vector (`c .* 2`, `abs.(c)`, `norm(c)`).
For all four, [`subband`](@ref)`(t, c, i)` is subband `i` as a matrix view, and
[`subband_layout`](@ref)`(t, size(img))` says which subband each `i` is.  For
the CT and NSCT the low-pass residual comes first, as
`(scale = J + 1, direction = 0)`, followed by the `2^L_array[j]` directional
subbands `(scale = j, direction = k)` of every scale `j`, finest first and
ordered by angle.  The TICT and STICT, described [below](@ref guide2d-tict),
have four channels per level.  For
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

img = Float64.(Gray.(testimage("barbara")))[1:512, 100:611]
t   = CT(J = 3, L_array = parabolic_levels(3))

println("direction exponents: ", t.L_array)   # 2^L subbands per scale
```

[`parabolic_levels`](@ref) applies the parabolic scaling rule (width ≈ length²),
doubling the direction count every other scale.

## Contourlet transform

```@example guide2d
coeffs = analyze(t, img)
rec    = synthesize(t, coeffs)
layout = subband_layout(t, size(img))

println("coarse size:        ", size(subband(t, coeffs, 1)))
println("subbands per scale: ", [count(s -> s.scale == j, layout) for j in 1:t.J])
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
function plot_coefficients(t, coeffs, dims; clip = 0.99)
    panels = Any[]
    for (i, s) in enumerate(subband_layout(t, dims))
        sb = subband(t, coeffs, i)
        if s.direction == 0     # the low-pass residual
            push!(panels, heatmap(sb; title = "coarse", color = :grays,
                aspect_ratio = :equal, axis = false, colorbar = false, yflip = true))
            continue
        end
        c = max(quantile(abs.(vec(sb)), clip), eps())
        push!(panels, heatmap(sb; title = "scale $(s.scale) dir $(s.direction)",
            color = :grays, clims = (-c, c), aspect_ratio = :equal, axis = false,
            colorbar = false, yflip = true))
    end
    cols = ceil(Int, sqrt(length(panels)))
    rows = ceil(Int, length(panels) / cols)
    return plot(panels...; layout = (rows, cols), size = (240 * cols, 240 * rows))
end

plot_coefficients(t, coeffs, size(img))
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
tn = NSCT(J = 2, L_array = [2, 3])

ns  = analyze(tn, img)
rec = synthesize(tn, ns)

println("coarse size: ", size(subband(tn, ns, 1)), " (input ", size(img), ")")
println("PR error:    ", maximum(abs, rec .- img))
@assert maximum(abs, rec .- img) < 2e-15
```

Shift invariance is a property the test suite enforces, not an approximation:

```@example guide2d
shift      = (3, 7)
ns_shifted = analyze(tn, circshift(img, shift))

err = maximum(  # every directional subband; subband 1 is the low-pass residual
    maximum(abs, subband(tn, ns_shifted, i) .- circshift(subband(tn, ns, i), shift))
        for i in 2:nsubbands(tn, size(img))
)
println("shift equivariance error: ", err)
@assert err < 1e-10
```

NSCT subbands are sparse — mostly near-zero with occasional strong edge
responses — so they are rendered with a lower clip (75th percentile) than the
critically sampled CT above:

```@example guide2d
plot_coefficients(tn, ns, size(img); clip = 0.75)
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

- [`TICT`](@ref) uses the nonsubsampled DFB: the whole transform is
  translation invariant.
- [`STICT`](@ref) uses the critically sampled DFB: only the pyramid is
  translation invariant, and the redundancy drops to `4J + 1`.

Channels 1 and 4 get the full directional tree; channels 2 and 3 pass mostly row
and mostly column frequencies, so they get half trees that split only that fan
(see [Theory](@ref theory-tict)).

```@example guide2d
tt  = TICT(J = 2, L_array = [3, 2])
tst = STICT(J = 2, L_array = [3, 2])

ti  = analyze(tt, img)
sti = analyze(tst, img)

ndirectional(t) = nsubbands(t, size(img)) - 1   # all but the low-pass residual
redundancy(c) = length(c) / length(img)

println("TICT:  ", ndirectional(tt), " subbands, redundancy ", redundancy(ti))
println("STICT: ", ndirectional(tst), " subbands, redundancy ", redundancy(sti))
println("PR errors: ", maximum(abs, synthesize(tt, ti) .- img), ", ",
    maximum(abs, synthesize(tst, sti) .- img))
```

[`subband_layout`](@ref) names their subbands `(level = j, channel = ch, direction = d)`,
after the low-pass residual `(level = J + 1, channel = 0, direction = 0)`: level by
level from the finest, channels 1 to 4 within a level, then the directions of
each channel.  Like
the NSCT, the TICT commutes with circular shifts exactly:

```@example guide2d
ti_shifted = analyze(tt, circshift(img, shift))
err = maximum(
    maximum(abs, subband(tt, ti_shifted, i) .- circshift(subband(tt, ti, i), shift))
        for i in 2:nsubbands(tt, size(img))
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

function plot_tict_level(t, c, j; clip = 0.99)
    layout = subband_layout(t, size(img))
    nd = maximum(s.direction for s in layout if s.level == j)
    panels = Any[]
    for ch in 1:4, d in 1:nd
        i = findfirst(==((level = j, channel = ch, direction = d)), layout)
        if i !== nothing
            sb = crop_center(subband(t, c, i))
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

plot_tict_level(tt, ti, 2)
savefig("tict_coeffs.png"); nothing # hide
```

![](tict_coeffs.png)

```@example guide2d
plot_tict_level(tst, sti, 2)
savefig("stict_coeffs.png"); nothing # hide
```

![](stict_coeffs.png)

Both transforms are FFT-based and have workspaces —
[`TICTWorkspace`](@ref), [`STICTWorkspace`](@ref), built by
[`make_workspace`](@ref) — with which the `!` forms allocate nothing.

## Choosing an API variant

Every transform comes in four forms. They compute the same thing; they differ
only in who owns the output container and the scratch buffers.

| Form | Output container | Scratch buffers | Use when |
|---|---|---|---|
| `analyze(t, img)` | allocated | allocated | one-shot processing |
| `analyze!(c, t, img)` | caller | allocated | you control the output's lifetime |
| `analyze(t, img; workspace = ws)` | allocated | reused | repeated calls, output kept per call |
| `analyze!(c, t, img; workspace = ws)` | caller | reused | iterative algorithms — no per-call allocation |

The synthesis has the same four forms: `synthesize(t, c)` and
`synthesize!(img_out, t, c)`, each with or without `workspace = ws`.  The `!`
and non-`!` entry points take identical arguments; the only difference is the
leading destination.  A workspace knows its transform (`transform(ws)`), so
`analyze(ws, img)`, `analyze!(c, ws, img)`, `synthesize(ws, c)` and
`synthesize!(img_out, ws, c)` are the same as the workspace forms above
without repeating `t`.

[`make_workspace`](@ref)`(t, img)` builds the scratch for images like `img`
(element type, size and array type, so a device array gives a device
workspace), and `make_workspace(t, Float64, (512, 512))` builds it from an
element type and a size.  A workspace serves one transform, element type and
image size; passing it with any other throws.  It runs at the precision of its
data: `Float32` images get a `Float32` workspace even with the default
`Float64` filters.  [`similar_coefficients`](@ref)`(t, img)` builds the
coefficient container, and `Base.summarysize(ws)` reports a workspace's actual
memory footprint.

```@example guide2d
ws         = make_workspace(tn, img)   # prewarm = true by default
coeffs_buf = similar_coefficients(tn, img)
img_out    = similar(img)

for _ in 1:10
    analyze!(coeffs_buf, tn, img; workspace = ws)

    # … threshold / prox step on coeffs_buf …

    synthesize!(img_out, tn, coeffs_buf; workspace = ws)
end

println("PR error after 10 iterations: ", maximum(abs, img_out .- img))
```

The NSCT workspace is where this matters most: it caches per-scale FFT plans and
precomputed filter spectra, which turns each call into an FFT, a pointwise
multiply and an inverse FFT. See [Performance](@ref performance) for what that
buys.

!!! note "Threading"
    A host NSCT or TICT workspace keeps one FFT buffer per Julia thread, and with
    threading on (`threaded = true`, or the default for complex data) the
    workspace forms split their independent subband FFTs between the threads.  The
    choice is made per call.

## Using the stages on their own

The pyramid and the directional filter bank are public, so a single stage can be
used without the full transform.  They are not exported; load them by name:

```@example guide2d
using Contourlets: lp_decompose, lp_reconstruct, dfb_decompose, dfb_reconstruct

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

[`nsdfb_decompose`](@ref Contourlets.nsdfb_decompose) / [`nsdfb_reconstruct`](@ref Contourlets.nsdfb_reconstruct) are the nonsubsampled
counterparts. The complete list is in the [API Reference](@ref api).
