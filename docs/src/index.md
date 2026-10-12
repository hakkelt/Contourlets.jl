# Contourlets.jl

*Directional multiresolution transforms for images in Julia: the Contourlet
Transform, the Nonsubsampled Contourlet Transform, and the translation-invariant
and semi-translation-invariant contourlet transforms.*

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/hakkelt/Contourlets.jl")
```

## Quick start

```julia
using Contourlets

img    = randn(256, 256)
params = ContourletParams(J = 4, L_array = parabolic_levels(4))

coeffs = ct_forward(img, params)        # critically sampled
rec    = ct_inverse(coeffs, params)

ns     = nsct_forward(img, params)      # shift-invariant, redundant
ns_rec = nsct_inverse(ns, params)
```

For iterative algorithms, preallocate the scratch buffers and the output once:

```julia
ws  = make_workspace(params, size(img))
buf = similar_coefficients(params, size(img))
out = similar(img)

for _ in 1:100
    ct_forward!(buf, img, params; workspace = ws)
    # … threshold / prox step on buf.subbands …
    ct_inverse!(out, buf, params; workspace = ws)
end
```

Volumes are covered by a separate package,
[Surfacelets.jl](https://github.com/hakkelt/Surfacelets.jl). It is not this
transform with an extra axis: the quincunx lattice, the shear index remap and
the Q2345 ladder are all planar, and Lu and Do prove that the hourglass
frequency partition a volume needs cannot be critically sampled at all. The two
packages share only the pyramid filter machinery, which lives in
[MultiscaleFilterBanks.jl](https://github.com/hakkelt/MultiscaleFilterBanks.jl).

## What is here

| Component | Description |
|-----------|-------------|
| [`ContourletParams`](@ref) | Scales `J`, direction counts `L_array`, filter pair |
| [`ct_forward`](@ref) / [`ct_inverse`](@ref) | Contourlet transform, critically sampled |
| [`nsct_forward`](@ref) / [`nsct_inverse`](@ref) | Nonsubsampled contourlet transform, shift-invariant |
| [`make_workspace`](@ref) / [`make_nsct_workspace`](@ref) | Preallocated buffers for iterative algorithms |
| [`parabolic_levels`](@ref) | Direction counts following the parabolic scaling rule |
| [`CDF97`](@ref) / [`Q2345`](@ref) | Default pyramid and ladder DFB filter pairs |

Threading is chosen per element type by default — SIMD for real data,
multi-threaded batching for complex — and the whole transform runs on the
GPU when a KernelAbstractions backend is loaded. Both are covered in
[Performance, threading and GPUs](@ref performance).

## Where to go next

- [2-D transforms](@ref guide2d) — CT, NSCT, TICT, STICT, the four API variants, and the
  pyramid and filter-bank stages on their own.
- [Denoising and approximation](@ref denoising) — worked comparisons against a
  separable wavelet.
- [Performance, threading and GPUs](@ref performance).
- [Theory](@ref theory) — the construction and where it leaves the papers.
- [Comparison with the MATLAB toolboxes](@ref comparison) — feature, accuracy
  and timing comparison against the reference implementations.
- [API Reference](@ref api).
