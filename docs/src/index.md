# Contourlets.jl

*Directional multiresolution transforms for images in Julia: the Contourlet
Transform, the Nonsubsampled Contourlet Transform, and the translation-invariant
and semi-translation-invariant contourlet transforms.*

## Installation

```julia
using Pkg
Pkg.add("Contourlets")
```

## Quick start

```julia
using Contourlets

img    = randn(256, 256)
t      = CT(scales = 4, direction_levels = parabolic_levels(4))
tn     = NSCT(scales = 4, direction_levels = parabolic_levels(4))

coeffs = analyze(t, img)                # critically sampled
rec    = synthesize(t, coeffs)

ns     = analyze(tn, img)               # shift-invariant, redundant
ns_rec = synthesize(tn, ns)
```

For iterative algorithms, preallocate the scratch buffers and the output once:

```julia
ws  = make_workspace(t, img)
buf = similar_coefficients(t, img)
out = similar(img)

for _ in 1:100
    analyze!(buf, t, img; workspace = ws)
    # … threshold / prox step on buf …
    synthesize!(out, t, buf; workspace = ws)
end
```

Volumes are covered by a separate package,
[Surfacelets.jl](https://github.com/hakkelt/Surfacelets.jl). It is not this
transform with an extra axis: the quincunx lattice, the shear index remap and
the Q2345 ladder are all planar, and Lu and Do prove that the hourglass
frequency partition a volume needs cannot be critically sampled at all. The two
packages share only the pyramid filter machinery, which lives in
[MultiscaleFilterBanks.jl](https://github.com/hakkelt/MultiscaleFilterBanks.jl).
[Shearlets.jl](https://github.com/hakkelt/Shearlets.jl) offers 2-D and 3-D
shearlet transforms built on the same core.

## What is here

| Component | Description |
|-----------|-------------|
| [`CT`](@ref) | Contourlet transform, critically sampled; each transform holds its number of scales `scales`, directional depths `direction_levels` and filters |
| [`NSCT`](@ref) | Nonsubsampled contourlet transform, shift-invariant |
| [`TICT`](@ref) | Translation-invariant CT (Eslami & Radha), shift-invariant |
| [`STICT`](@ref) | Semi-translation-invariant CT: invariant pyramid, decimated DFB, redundancy `4 scales + 1` |
| [`analyze`](@ref) / [`synthesize`](@ref) (and [`analyze!`](@ref) / [`synthesize!`](@ref)) | Run a transform |
| [`make_workspace`](@ref) / [`similar_coefficients`](@ref) | Preallocated buffers for iterative algorithms |
| [`subband_layout`](@ref) / [`subband`](@ref) | Which subband is which, and its coefficients |
| [`parabolic_levels`](@ref) | Direction counts following the parabolic scaling rule |
| [`CDF97`](@ref) / [`Q2345`](@ref) | Default pyramid and ladder DFB filter pairs |

Threading is chosen per element type by default — SIMD for real data,
multi-threaded batching for complex — and the NSCT and TICT take a `threaded`
keyword to override it (the CT and STICT accept it and ignore it). The whole transform runs on the GPU with CUDA.jl,
AMDGPU.jl or Metal.jl; OpenCL.jl is supported but much slower than the CPU,
because it has no FFT. Both are covered in [Performance, threading and
GPUs](@ref performance).

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
