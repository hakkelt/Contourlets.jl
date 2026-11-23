# Contourlets.jl

Contourlet transforms of images in Julia. A contourlet transform splits an
image by scale, like a wavelet transform, and splits each scale by
orientation, so that edges and oriented textures are represented by few
coefficients. The package provides the contourlet transform ([`CT`](@ref)), the
nonsubsampled contourlet transform ([`NSCT`](@ref)), and the translation-invariant
and semi-translation-invariant contourlet transforms ([`TICT`](@ref),
[`STICT`](@ref)).

## Installation

Contourlets.jl and its dependencies AbstractMultiscaleTransforms.jl and
MultiscaleFilterBanks.jl are not registered. Clone the three repositories side
by side:

```bash
git clone https://github.com/hakkelt/AbstractMultiscaleTransforms.jl AbstractMultiscaleTransforms
git clone https://github.com/hakkelt/MultiscaleFilterBanks.jl MultiscaleFilterBanks
git clone https://github.com/hakkelt/Contourlets.jl Contourlets
```

and add them to your environment:

```julia
using Pkg
Pkg.develop([PackageSpec(path = "AbstractMultiscaleTransforms"),
             PackageSpec(path = "MultiscaleFilterBanks"),
             PackageSpec(path = "Contourlets")])
```

## Example

```@example index
using Contourlets, Random

img = randn(Xoshiro(1), 256, 256)
t = NSCT(scales = 3, direction_levels = [2, 3, 3])
c = analyze(t, img)                 # 256 × 256 × 21: lowpass + 4 + 8 + 8 directions
subband_layout(t, size(img))[2]     # what subband 2 is
```

```@example index
maximum(abs, synthesize(t, c) .- img)
```

## Contents

How-to guides:

- [Choose a transform and find its subbands](@ref guide-transforms)
- [Reuse a workspace](@ref guide-workspaces) for repeated transforms
- [Denoise an image with the NSCT](@ref denoising)
- [Use threads and GPUs](@ref guide-parallel)
- [Use the pyramid and filter-bank stages alone](@ref guide-stages)

Explanation: [Theory](@ref theory) describes how the transforms are built and
lists the papers they come from.

Reference: the [API Reference](@ref api) documents every public function and
type.

Related packages: [Surfacelets.jl](https://github.com/hakkelt/Surfacelets.jl)
provides the surfacelet transform for volumes, and
[Shearlets.jl](https://github.com/hakkelt/Shearlets.jl) 2-D and 3-D shearlet
transforms. All of them implement the interface of
[AbstractMultiscaleTransforms.jl](https://github.com/hakkelt/AbstractMultiscaleTransforms.jl).
