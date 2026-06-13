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
using Contourlets, Wavelets, Random, Plots, TestImages, ImageCore, Colors
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

function nsct_local_variance_denoise(noisy, params, gains, σ; window_size=5)
    c = nsct_forward(noisy, params)
    pad = window_size ÷ 2
    for j in eachindex(c.subbands), kk in eachindex(c.subbands[j])
        sb = c.subbands[j][kk]
        σ_n = σ * gains[j][kk]
        var_n = σ_n^2
        out = similar(sb)
        R, C = size(sb)
        for c_idx in 1:C, r_idx in 1:R
            energy = 0.0
            count = 0
            for dc in -pad:pad, dr in -pad:pad
                r_w = clamp(r_idx + dr, 1, R)
                c_w = clamp(c_idx + dc, 1, C)
                energy += sb[r_w, c_w]^2
                count += 1
            end
            energy /= count
            
            σ_w = sqrt(max(0.0, energy - var_n))
            T = σ_w > 0.0 ? var_n / σ_w : var_n / 1e-6
            
            val = sb[r_idx, c_idx]
            out[r_idx, c_idx] = sign(val) * max(0.0, abs(val) - T)
        end
        c.subbands[j][kk] .= out
    end
    return nsct_inverse(c)
end

function nsct_bivariate_denoise(noisy, params, gains, σ; window_size=5)
    c = nsct_forward(noisy, params)
    pad = window_size ÷ 2
    for j in eachindex(c.subbands), kk in eachindex(c.subbands[j])
        sb = c.subbands[j][kk]
        σ_n = σ * gains[j][kk]
        var_n = σ_n^2
        
        parent_sb = if j == length(c.subbands)
            c.coarse
        else
            L_curr = length(c.subbands[j])
            L_parent = length(c.subbands[j+1])
            parent_kk = L_curr >= L_parent ? (kk - 1) ÷ (L_curr ÷ L_parent) + 1 : (kk - 1) * (L_parent ÷ L_curr) + 1
            c.subbands[j+1][parent_kk]
        end
        
        out = similar(sb)
        R, C = size(sb)
        for c_idx in 1:C, r_idx in 1:R
            energy = 0.0
            count = 0
            for dc in -pad:pad, dr in -pad:pad
                r_w = clamp(r_idx + dr, 1, R)
                c_w = clamp(c_idx + dc, 1, C)
                energy += sb[r_w, c_w]^2
                count += 1
            end
            energy /= count
            
            σ_w = sqrt(max(0.0, energy - var_n))
            
            y1 = sb[r_idx, c_idx]
            y2 = parent_sb[r_idx, c_idx]
            y = sqrt(y1^2 + y2^2)
            
            T = σ_w > 0.0 ? (sqrt(3.0) * var_n) / σ_w : (sqrt(3.0) * var_n) / 1e-6
            
            out[r_idx, c_idx] = y > T ? (y - T) / y * y1 : 0.0
        end
        c.subbands[j][kk] .= out
    end
    return nsct_inverse(c)
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

We corrupt a crop of the *Barbara* image (focusing on the fabric regions) with 
white Gaussian noise and evaluate three different thresholding strategies:

1. **Hard Thresholding**: A universal threshold ``\lambda = k\,\sigma\,\gamma_{j,k}`` is applied independently, where ``\gamma_{j,k}`` is the estimated noise gain of the subband.
2. **Local Variance-Adaptive Thresholding**: A spatially-adaptive soft threshold. We estimate the local signal variance ``\sigma_w^2 = \max(0, \frac{1}{N} \sum y_i^2 - \sigma_n^2)`` in a small window. The threshold is dynamically set as ``T = \sigma_n^2 / \sigma_w``.
3. **Bivariate Shrinkage**: Proposed by Sendur & Selesnick, this models the dependency between a coefficient ``y_1`` and its parent coefficient ``y_2`` (at the same spatial location in the next coarser scale). The joint magnitude is ``y = \sqrt{y_1^2 + y_2^2}``, and the shrinkage function is:
   ```math
   \hat{w}_1 = \frac{(y - \sqrt{3}\sigma_n^2 / \sigma_w)_{+}}{y} y_1
   ```
   This heavily shrinks coefficients whose parents are small (likely noise), while preserving true edges that persist across multiple scales.

```@example nla
# Use a 512x512 crop of the Barbara image containing oriented textures
ref = reverse(Float64.(Gray.(testimage("barbara")))[1:512, 100:611], dims=2)
shading = ref

Random.seed!(11)
σ = 0.03
noisy = shading .+ σ .* randn(size(shading)...)

gains = nsct_noise_gains(params, size(shading))
ct_rec = nsct_denoise(noisy, params, gains, σ, 3.0)
ct_local = nsct_local_variance_denoise(noisy, params, gains, σ)
ct_bivar = nsct_bivariate_denoise(noisy, params, gains, σ)
wt_rec = wt_denoise(noisy, 3σ)

println("noisy            : $(round(psnr(shading, noisy);  digits = 2)) dB")
println("9/7 wavelet hard : $(round(psnr(shading, wt_rec); digits = 2)) dB")
println("NSCT hard        : $(round(psnr(shading, ct_rec); digits = 2)) dB")
println("NSCT local var   : $(round(psnr(shading, ct_local); digits = 2)) dB")
println("NSCT bivariate   : $(round(psnr(shading, ct_bivar); digits = 2)) dB")
println("Bivariate gain over wavelet: $(round(psnr(shading, ct_bivar) - psnr(shading, wt_rec); digits = 2)) dB")
```

```@example nla
p2 = plot(layout = (2, 2), size = (800, 800))
heatmap!(p2[1], noisy, title = "Noisy", color = :grays, aspect_ratio = :equal, axis = false, colorbar = false)
heatmap!(p2[2], wt_rec, title = "Wavelet hard", color = :grays, aspect_ratio = :equal, axis = false, colorbar = false)
heatmap!(p2[3], ct_rec, title = "NSCT hard", color = :grays, aspect_ratio = :equal, axis = false, colorbar = false)
heatmap!(p2[4], ct_bivar, title = "NSCT bivariate", color = :grays, aspect_ratio = :equal, axis = false, colorbar = false)
savefig(p2, "nla_denoise.svg"); nothing # hide
```

![](nla_denoise.svg)

```@example nla
p3 = plot(layout = (2, 2), size = (800, 800))
heatmap!(p3[1], abs.(shading .- noisy), title = "Noisy (Diff)", color = :grays, aspect_ratio = :equal, axis = false, colorbar = false)
heatmap!(p3[2], abs.(shading .- wt_rec), title = "Wavelet hard (Diff)", color = :grays, aspect_ratio = :equal, axis = false, colorbar = false)
heatmap!(p3[3], abs.(shading .- ct_rec), title = "NSCT hard (Diff)", color = :grays, aspect_ratio = :equal, axis = false, colorbar = false)
heatmap!(p3[4], abs.(shading .- ct_bivar), title = "NSCT bivariate (Diff)", color = :grays, aspect_ratio = :equal, axis = false, colorbar = false)
savefig(p3, "nla_diff.svg"); nothing # hide
```

![](nla_diff.svg)

!!! note "Scope and reproducibility"
    These are small (128²) synthetic experiments meant to illustrate the
    *directional* benefit; the exact gains depend on the image, parameters and
    threshold.  On smooth or piecewise-constant images the critically sampled
    wavelet is hard to beat — the contourlet advantage appears specifically for
    oriented texture and long smooth contours.  Larger gains on natural images
    (the ≈1.46 dB reported by Do & Vetterli 2005) use statistically tuned
    thresholds (e.g. the hidden-Markov-tree model of Po & Do 2006) rather than
    the single universal threshold used here.
