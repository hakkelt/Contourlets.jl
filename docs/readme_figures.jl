# Writes the two README figures to docs/src/assets:
#
#     julia --project=docs docs/readme_figures.jl

using Contourlets, FFTW, ImageCore, Plots, Random, Statistics, TestImages

const ASSETS = joinpath(@__DIR__, "src", "assets")

# Frequency tiling of a CT: at each frequency, the largest response of any subband.
function tiling_figure()
    N = 128
    t = CT(scales = 2, direction_levels = [2, 3])
    tiling = zeros(N, N)
    for i in 1:nsubbands(t, (N, N))
        c = analyze(t, zeros(N, N))
        sb = subband(t, c, i)
        sb[size(sb, 1) ÷ 2 + 1, size(sb, 2) ÷ 2 + 1] = 1.0
        tiling .= max.(tiling, abs.(fftshift(fft(synthesize(t, c)))))
    end
    return heatmap(
        log1p.(tiling); color = :viridis, aspect_ratio = :equal,
        axis = false, grid = false, colorbar = false, size = (300, 300), margin = 0Plots.mm
    )
end

# A noisy crop of Barbara and its NSCT hard-threshold estimate.
function denoising_figure()
    ref = Float64.(Gray.(testimage("barbara")))[1:512, 100:611]
    σ = 0.08
    noisy = ref .+ σ .* randn(Xoshiro(11), size(ref))
    t = NSCT(scales = 3, direction_levels = [3, 3, 3])
    ws = make_workspace(t, ref)
    noise = analyze(t, randn(Xoshiro(2), size(ref)); workspace = ws)
    c = analyze(t, noisy; workspace = ws)
    for i in 2:nsubbands(t, size(ref))
        sb = subband(t, c, i)
        λ = 3σ * std(subband(t, noise, i))
        sb .= ifelse.(abs.(sb) .>= λ, sb, 0)
    end
    denoised = synthesize(t, c; workspace = ws)
    psnr(x) = round(10 * log10(1 / mean(abs2, x .- ref)); digits = 1)
    gray(x, title) = heatmap(
        x[129:384, 129:384]; title, titlefontsize = 10,
        color = :grays, clims = (0, 1), aspect_ratio = :equal, axis = false,
        grid = false, colorbar = false, yflip = true
    )
    return plot(
        gray(noisy, "noisy, $(psnr(noisy)) dB"),
        gray(denoised, "NSCT, $(psnr(denoised)) dB"); layout = (1, 2), size = (600, 300)
    )
end

savefig(tiling_figure(), joinpath(ASSETS, "readme-tiling.png"))
savefig(denoising_figure(), joinpath(ASSETS, "readme-denoising.png"))
