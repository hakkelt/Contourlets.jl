# Contourlets.jl

A Julia package implementing the Discrete Contourlet Transform (CT) and
Nonsubsampled Contourlet Transform (NSCT) via a Pyramid Directional Filter Bank (PDFB).

## Features

- **Laplacian Pyramid** multiscale decomposition with CDF 9/7 filters
- **Directional Filter Bank** (DFB) binary-tree decomposition with the "23-45"
  Phoong et al. (1995) ladder filters
- **CT**: `ct_forward` / `ct_inverse` — nearly critically sampled, O(N) complexity
- **NSCT**: `nsct_forward` / `nsct_inverse` — invariant under circular shifts via
  à trous upsampling
- **Workspace API**: reuse preallocated buffers across iterations with
  `make_workspace` / `ct_forward!`
- Parabolic scaling utility `parabolic_levels` for optimal direction count per scale

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/hakkelt/Contourlets.jl")
```

## Quick Start

```julia
using Contourlets, Random

img = randn(256, 256)
params = ContourletParams(J=4, L_array=parabolic_levels(4))

# Allocating API
coeffs = ct_forward(img, params)
img_rec = ct_inverse(coeffs)
@assert maximum(abs, img_rec .- img) < 1e-12

# Preallocated-buffer API (for iterative algorithms)
ws = make_workspace(params, size(img))
coeffs = similar_coefficients(params, size(img))
for iter in 1:100
    ct_forward!(coeffs, img, ws)
    # … threshold / prox step on coeffs …
    ct_inverse!(img, coeffs, ws)
end
```

The `!` entry points reuse the workspace's buffers for the pyramid stage; the
directional stage still allocates its subband tree, so this reduces — not fully
eliminates — per-iteration allocation.

## Development

This library was implemented with the assistance of [GitHub Copilot](https://github.com/features/copilot) and [Anthropic Claude SDK](https://docs.anthropic.com/en/api/getting-started).

## References

- Do & Vetterli (2005), "The Contourlet Transform: An Efficient Directional Multiresolution Image Representation"
- da Cunha, Zhou & Do (2006), "The Nonsubsampled Contourlet Transform"
- Phoong, Kim, Vaidyanathan & Ansari (1995), "A New Class of Two-Channel Biorthogonal Filter Banks"
