# [Nonlinear Approximation & Denoising](@id nla)

This page reproduces the two experiments that motivate the contourlet transform
(Do & Vetterli 2005, §VI) and compares it against a separable 2-D "9/7" wavelet
from [Wavelets.jl](https://github.com/JuliaDSP/Wavelets.jl):

1. **M-term nonlinear approximation (NLA)** — keep only the `M` most significant
   coefficients and reconstruct.  Contourlets represent oriented texture with
   *directional* basis elements, which separable wavelets cannot.
2. **Denoising** — hard-threshold the coefficients of a noisy image.  The
   shift-invariant NSCT removes noise from oriented features with less ringing.

Both transforms use the same CDF 9/7 filters, so the comparison isolates the
*directional* contribution.  The numbers below are computed live during the docs
build (so they will track the implementation).

```@setup nla
using Contourlets, Wavelets, Random, Plots
Plots.default(show = false)

psnr(ref, x) = 10 * log10((maximum(ref) - minimum(ref))^2 / (sum(abs2, x .- ref) / length(ref)))

const WT97 = wavelet(WT.cdf97, WT.Lifting)

# Per-subband synthesis gain γ — the L2 norm contributed to the reconstruction
# by a unit coefficient in each directional subband.  Weighting coefficient
# magnitudes by γ makes the M-term selection energy-consistent across the
# (non-orthonormal, redundant) contourlet frame.
function ct_subband_gains(params, sz)
    template = ct_forward(zeros(sz...), params)
    γ = [zeros(length(sbs)) for sbs in template.subbands]
    for j in eachindex(template.subbands), k in eachindex(template.subbands[j])
        c = ct_forward(zeros(sz...), params)
        c.subbands[j][k] .= randn(size(c.subbands[j][k]))
        γ[j][k] = sqrt(sum(abs2, ct_inverse(c)) / length(c.subbands[j][k]))
    end
    return γ
end

# Energy-normalised M-term contourlet approximation.
function ct_nla(img, params, γ, M)
    c = ct_forward(img, params)
    scaled = reduce(vcat, [abs.(vec(c.subbands[j][k])) .* γ[j][k]
        for j in eachindex(c.subbands) for k in eachindex(c.subbands[j])])
    thr = M < length(scaled) ? partialsort(scaled, M; rev = true) : -1.0
    for j in eachindex(c.subbands), k in eachindex(c.subbands[j])
        sb = c.subbands[j][k]
        @. sb = ifelse(abs(sb) * γ[j][k] >= thr, sb, 0.0)
    end
    return ct_inverse(c)
end

# M-term wavelet approximation (keep the M largest dwt coefficients).
function wt_nla(img, M)
    c = dwt(img, WT97)
    v = sort!(abs.(vec(c)); rev = true)
    thr = M < length(v) ? v[M] : -1.0
    @. c = ifelse(abs(c) >= thr, c, 0.0)
    return idwt(c, WT97)
end

# Per-subband noise standard deviation of the NSCT for unit-variance input
# noise — the correct scale for the universal hard threshold k·σ·gain.
function nsct_noise_gains(params, sz; trials = 6)
    acc = nothing
    for _ in 1:trials
        c = nsct_forward(randn(sz...), params)
        g = [[sqrt(sum(abs2, sb) / length(sb)) for sb in sbs] for sbs in c.subbands]
        acc = acc === nothing ? g : [acc[j] .+ g[j] for j in eachindex(g)]
    end
    return [acc[j] ./ trials for j in eachindex(acc)]
end

function nsct_denoise(noisy, params, gains, σ, k)
    c = nsct_forward(noisy, params)
    for j in eachindex(c.subbands), kk in eachindex(c.subbands[j])
        λ = k * σ * gains[j][kk]
        sb = c.subbands[j][kk]
        @. sb = ifelse(abs(sb) >= λ, sb, 0.0)
    end
    return nsct_inverse(c)
end

function wt_denoise(noisy, λ)
    c = dwt(noisy, WT97)
    @. c = ifelse(abs(c) >= λ, c, 0.0)
    return idwt(c, WT97)
end
```

## M-term approximation of directional texture

Separable wavelets struggle with oriented oscillations.  We build an image of
four texture patches, each at a different orientation — the structure
contourlets are designed for.

```@example nla
using Contourlets, Wavelets, Random, Plots

N = 128
texture = zeros(Float64, N, N)
for i in 1:N, j in 1:N
    θ = (i <= N ÷ 2 ? (j <= N ÷ 2 ? 20 : 75) : (j <= N ÷ 2 ? 110 : 160)) * π / 180
    texture[i, j] = sin(2π * 0.3 * (cos(θ) * i + sin(θ) * j))
end
heatmap(texture, color = :grays, title = "Directional texture", aspect_ratio = :equal,
    axis = false, colorbar = false)
savefig("nla_texture.svg"); nothing # hide
```

![](nla_texture.svg)

Keeping the same number of coefficients `M` for both transforms:

```@example nla
params = ContourletParams(J = 3, L_array = [3, 3, 2])
γ = ct_subband_gains(params, (N, N))

for M in (512, 1024, 2048)
    ct = psnr(texture, ct_nla(texture, params, γ, M))
    wv = psnr(texture, wt_nla(texture, M))
    println("M = $(lpad(M, 4)):  contourlet $(round(ct; digits = 2)) dB   " *
            "9/7 wavelet $(round(wv; digits = 2)) dB   " *
            "gain $(round(ct - wv; digits = 2)) dB")
end
```

The contourlet keeps a higher PSNR at every budget: its directional subbands
match the texture orientation, so fewer coefficients describe each patch.

```@example nla
M = 2048
ct_approx = ct_nla(texture, params, γ, M)
wt_approx = wt_nla(texture, M)
p = plot(layout = (1, 3), size = (900, 320))
heatmap!(p[1], texture, title = "Original", color = :grays, aspect_ratio = :equal, axis = false, colorbar = false)
heatmap!(p[2], ct_approx, title = "Contourlet ($M terms)", color = :grays, aspect_ratio = :equal, axis = false, colorbar = false)
heatmap!(p[3], wt_approx, title = "Wavelet ($M terms)", color = :grays, aspect_ratio = :equal, axis = false, colorbar = false)
savefig(p, "nla_mterm.svg"); nothing # hide
```

![](nla_mterm.svg)

## Denoising with the shift-invariant NSCT

We corrupt a "texture + smooth shading" image (a stand-in for the fabric regions
of *Barbara*) with white Gaussian noise and hard-threshold each NSCT subband at
its own noise level ``k\,\sigma\,\gamma_{j,k}``, estimated by passing unit
noise through the transform.

```@example nla
shading = zeros(Float64, N, N)
for i in 1:N, j in 1:N
    base = 0.5 + 0.3 * sin(2π * (i + j) / 180)
    tex = (40 <= i <= 100 && 30 <= j <= 95) ? 0.35 * sin(2π * 0.25 * (0.8i + 0.6j)) : 0.0
    shading[i, j] = base + tex
end

Random.seed!(11)
σ = 0.1
noisy = shading .+ σ .* randn(N, N)

gains = nsct_noise_gains(params, (N, N))
ct_rec = nsct_denoise(noisy, params, gains, σ, 3.0)
wt_rec = wt_denoise(noisy, 3σ)

println("noisy      : $(round(psnr(shading, noisy);  digits = 2)) dB")
println("9/7 wavelet: $(round(psnr(shading, wt_rec); digits = 2)) dB")
println("NSCT       : $(round(psnr(shading, ct_rec); digits = 2)) dB")
println("NSCT gain over wavelet: $(round(psnr(shading, ct_rec) - psnr(shading, wt_rec); digits = 2)) dB")
```

```@example nla
p2 = plot(layout = (1, 3), size = (900, 320))
heatmap!(p2[1], noisy, title = "Noisy", color = :grays, aspect_ratio = :equal, axis = false, colorbar = false)
heatmap!(p2[2], ct_rec, title = "NSCT denoised", color = :grays, aspect_ratio = :equal, axis = false, colorbar = false)
heatmap!(p2[3], wt_rec, title = "Wavelet denoised", color = :grays, aspect_ratio = :equal, axis = false, colorbar = false)
savefig(p2, "nla_denoise.svg"); nothing # hide
```

![](nla_denoise.svg)

!!! note "Scope and reproducibility"
    These are small (128²) synthetic experiments meant to illustrate the
    *directional* benefit; the exact gains depend on the image, parameters and
    threshold.  On smooth or piecewise-constant images the critically sampled
    wavelet is hard to beat — the contourlet advantage appears specifically for
    oriented texture and long smooth contours.  Larger gains on natural images
    (the ≈1.46 dB reported by Do & Vetterli 2005) use statistically tuned
    thresholds (e.g. the hidden-Markov-tree model of Po & Do 2006) rather than
    the single universal threshold used here.
