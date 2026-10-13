# Contourlets.jl

[![CI](https://github.com/hakkelt/Contourlets.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/hakkelt/Contourlets.jl/actions/workflows/CI.yml)
[![Documentation (dev)](https://img.shields.io/badge/docs-dev-blue.svg)](https://hakkelt.github.io/Contourlets.jl/dev/)
[![Coverage](https://codecov.io/gh/hakkelt/Contourlets.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/hakkelt/Contourlets.jl)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![JET](https://img.shields.io/badge/%F0%9F%9B%A9%EF%B8%8F_tested_with-JET.jl-233f9a)](https://github.com/aviatesk/JET.jl)
[![Runic](https://img.shields.io/badge/code%20style-runic-blue.svg)](https://github.com/fredrikekre/Runic.jl)

A Julia package implementing the Discrete Contourlet Transform (CT), the
Nonsubsampled Contourlet Transform (NSCT) and the translation-invariant (TICT)
and semi-translation-invariant (STICT) contourlet transforms via a Pyramid
Directional Filter Bank (PDFB).

| Contourlet frequency tiling | NSCT denoising vs. wavelets |
|:---:|:---:|
| [![Contourlet frequency tiling](https://hakkelt.github.io/Contourlets.jl/dev/ct_tiling.png)](https://hakkelt.github.io/Contourlets.jl/dev/theory/) | [![NSCT denoising](https://hakkelt.github.io/Contourlets.jl/dev/guide/nla_denoise.png)](https://hakkelt.github.io/Contourlets.jl/dev/guide/denoising/) |
| Directional wedges tile the frequency plane ([Theory](https://hakkelt.github.io/Contourlets.jl/dev/theory/)). | NSCT recovers edges a separable wavelet smears ([Denoising](https://hakkelt.github.io/Contourlets.jl/dev/guide/denoising/)). |

## Features

- **Laplacian Pyramid** multiscale decomposition with CDF 9/7 filters
- **Directional Filter Bank** (DFB) with the "23-45" Phoong et al. (1995) ladder
  filters, computed as the multichannel bank of Do's thesis (closed-form
  equivalent filters on the DFT grid, diagonal sampling per channel)
- **CT**: `ct_forward` / `ct_inverse` — nearly critically sampled, O(N) complexity
- **NSCT**: `nsct_forward` / `nsct_inverse` — invariant under circular shifts via
  à trous upsampling
- **TICT / STICT**: `tict_forward` / `stict_forward` (Eslami & Radha 2006) — a
  translation-invariant Laplacian pyramid with nonsubsampled or critically
  sampled directional filter banks
- **Workspace API**: reuse preallocated buffers across iterations with
  `make_workspace` and the `workspace=ws` kwarg on both `ct_forward!` and `ct_forward`
- Parabolic scaling utility `parabolic_levels` for optimal direction count per scale
- **GPU**: the full transform runs on device-resident arrays for any
  KernelAbstractions backend; CUDA additionally captures the workspace paths
  into CUDA graphs

## Quick Start

```julia
using Contourlets, Random

img = randn(256, 256)
params = ContourletParams(J=4, L_array=parabolic_levels(4))

# Allocating API
coeffs = ct_forward(img, params)
img_rec = ct_inverse(coeffs, params)
@assert maximum(abs, img_rec .- img) < 1e-12

# Preallocated-buffer API (for iterative algorithms)
ws = make_workspace(params, size(img))
coeffs = similar_coefficients(params, size(img))
img_out = similar(img)
for iter in 1:100
    ct_forward!(coeffs, img, params; workspace=ws)
    # … threshold / prox step on coeffs …
    ct_inverse!(img_out, coeffs, params; workspace=ws)
end
```

The `!` entry points accept the same `params` positional and optional `workspace`
kwarg as the allocating forms; the only difference is that `!` writes into a
caller-supplied output container instead of allocating one.  With `workspace=ws`
the pyramid stage reuses preallocated scratch buffers — no per-iteration
allocation.

## Volumes

Volumes are covered by [Surfacelets.jl](https://github.com/hakkelt/Surfacelets.jl).
It is **not** this transform with an extra axis: the quincunx lattice, the shear
index remap and the Q2345 ladder are all planar, and Lu and Do prove that the
hourglass frequency partition a volume needs cannot be critically sampled at all,
so the surfacelet transform is a different construction.  The two packages share
only the pyramid filter machinery, which lives in
[MultiscaleFilterBanks.jl](https://github.com/hakkelt/MultiscaleFilterBanks.jl)
and is re-exported here.

To work against a local checkout of the core package instead, put it next to
this one and run `julia --project=. dev/bootstrap.jl` once; it
`Pkg.develop`s the sibling checkout into every environment of the workspace.

## References

- Do & Vetterli (2005), "The Contourlet Transform: An Efficient Directional Multiresolution Image Representation"
- da Cunha, Zhou & Do (2006), "The Nonsubsampled Contourlet Transform"
- Phoong, Kim, Vaidyanathan & Ansari (1995), "A New Class of Two-Channel Biorthogonal Filter Banks"

## Acknowledgments

The transforms are implemented from the papers that introduced them: the
contourlet transform of Do and Vetterli (2005), the nonsubsampled contourlet
transform of da Cunha, Zhou and Do (2006), the translation-invariant and
semi-translation-invariant contourlet transforms of Eslami and Radha (2006), the
directional filter bank of
Bamberger and Smith (1992) as formulated in Do's thesis (2001), the quincunx
filters of Phoong, Kim, Vaidyanathan and Ansari (1995), and the Laplacian pyramid
of Burt and Adelson (1983).  The [Theory](https://hakkelt.github.io/Contourlets.jl/dev/theory/)
page gives the full references.

The authors' MATLAB toolboxes —
[Contourlet Toolbox](https://www.mathworks.com/matlabcentral/fileexchange/8837-contourlet-toolbox)
(GPL) and
[Nonsubsampled Contourlet Toolbox](https://www.mathworks.com/matlabcentral/fileexchange/10049-nonsubsampled-contourlet-toolbox)
(no licence stated) — served as black-box references for validation and
benchmarks only; no code is derived from them.  See [`LICENSE`](LICENSE).

## Disclaimer

This library was created with the assistance of [Claude](https://claude.com/claude-code),
[Antigravity](https://antigravity.google/), and
[GitHub Copilot](https://github.com/features/copilot).
