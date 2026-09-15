# Contourlets.jl

*Directional multiresolution transforms in Julia: the Contourlet Transform, the
Nonsubsampled Contourlet Transform, and their 3-D counterparts — the 3-D NSCT
and the decimated surfacelet transform.*

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

Volumes use a parallel 3-D API, including a streaming form for sizes where the
coefficient set does not fit in memory:

```julia
vol     = randn(Float32, 256, 256, 256)
params3 = ContourletParams3D(J = 3, L_array = [2, 2, 3])   # 48 directional subbands
ws3     = make_nsct3d_workspace(params3, size(vol), Float32)

coarse = nsct3d_forward(vol, params3; workspace = ws3) do j, k, band
    # `band` is valid only until this callback returns
end
```

The decimated counterpart, [`surfacelet3d_forward`](@ref), stores every subband
at its own reduced resolution — about 3× leaner at useful settings, at the cost
of shift invariance.

## What is here

| Component | Description |
|-----------|-------------|
| [`ContourletParams`](@ref) | Scales `J`, direction counts `L_array`, filter pair |
| [`ct_forward`](@ref) / [`ct_inverse`](@ref) | Contourlet transform, critically sampled |
| [`nsct_forward`](@ref) / [`nsct_inverse`](@ref) | Nonsubsampled contourlet transform, shift-invariant |
| [`make_workspace`](@ref) / [`make_nsct_workspace`](@ref) | Preallocated buffers for iterative algorithms |
| [`parabolic_levels`](@ref) | Direction counts following the parabolic scaling rule |
| [`CDF97`](@ref) / [`Q2345`](@ref) | Default pyramid and ladder DFB filter pairs |
| [`ContourletParams3D`](@ref) | 3-D parameters: scales, per-scale `L`, Bernstein order |
| [`nsct3d_forward`](@ref) / [`nsct3d_inverse`](@ref) | 3-D NSCT, with streaming forms |
| [`make_nsct3d_workspace`](@ref) | Preallocated 3-D plans and buffers |
| [`make_nsdfb3d`](@ref) | 3-D directional filter bank alone (`3·2^L` subbands) |
| [`SurfaceletParams3D`](@ref) | Decimated 3-D transform: scales, per-scale `L`, transition widths |
| [`surfacelet3d_forward`](@ref) / [`surfacelet3d_inverse`](@ref) | Surfacelet transform, subbands at reduced resolution |
| [`make_surfacelet3d`](@ref) | Plan the surfacelet masks and FFTs for one volume size |

Threading is chosen per element type by default — SIMD for real data,
multi-threaded batching for complex — and the whole 2-D transform runs on the
GPU when a KernelAbstractions backend is loaded. Both are covered in
[Performance, threading and GPUs](@ref performance).

## Where to go next

- [2-D transforms](@ref guide2d) — CT, NSCT, the four API variants, and the
  pyramid and filter-bank stages on their own.
- [3-D transforms](@ref guide3d) — the 3-D NSCT, streaming, and surfacelets.
- [Denoising and approximation](@ref denoising) — worked comparisons against a
  separable wavelet.
- [Performance, threading and GPUs](@ref performance).
- [Theory](@ref theory) and [Theory — 3-D](@ref theory3d) — the constructions
  and where they leave the papers.
- [Comparison with the MATLAB toolboxes](@ref comparison) — feature, accuracy
  and timing comparison against the reference implementations.
- [API Reference](@ref api).
