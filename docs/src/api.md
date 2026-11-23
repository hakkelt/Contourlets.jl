# [API Reference](@id api)

```@meta
CurrentModule = Contourlets
```

The transforms, their workspaces and the interface functions are exported. The
pyramid and directional-filter-bank stages under [Building blocks](@ref
building-blocks) are public but not exported: import them by name, as in
`using Contourlets: lp_decompose`.

## Module

```@docs
Contourlets
```

## Transforms

```@docs
CT
NSCT
TICT
STICT
parabolic_levels
```

## Running a transform

These functions come from
[AbstractMultiscaleTransforms.jl](https://github.com/hakkelt/AbstractMultiscaleTransforms.jl);
each entry gives the general contract first and the contourlet methods after
it. Every function also accepts a workspace in place of the transform, as in
`analyze(ws, x)` and `synthesize!(x, ws, c)`.

```@docs
analyze
analyze!
synthesize
synthesize!
```

## Workspaces

```@docs
make_workspace
CTWorkspace
NSCTWorkspace
TICTWorkspace
STICTWorkspace
transform
```

## Coefficients

```@docs
similar_coefficients
subband_layout
nsubbands
subband
SubbandVector
SubbandLayout
```

## Filters

```@docs
CDF97
Q2345
FilterPair
QuincunxFilterPair
upsample_kernel
```

## [Building blocks](@id building-blocks)

### Pyramids

```@docs
lp_decompose
lp_decompose!
lp_reconstruct
lp_reconstruct!
nsp_decompose
nsp_decompose!
nsp_reconstruct
nsp_reconstruct!
```

### Directional filter banks

```@docs
dfb_decompose
dfb_decompose!
dfb_reconstruct
dfb_reconstruct!
dfb_subband_sizes
nsdfb_decompose
nsdfb_reconstruct
qfb_decompose
qfb_decompose!
qfb_reconstruct
qfb_reconstruct!
```

## Interface types

Types and functions of AbstractMultiscaleTransforms.jl and MultiscaleFilterBanks.jl
that the entries above refer to.

```@docs
AbstractMultiscaleTransform
AbstractTransformWorkspace
AbstractMultiscaleTransforms.DefaultWorkspace
AbstractMultiscaleTransforms.check_workspace
AbstractMultiscaleTransforms.supports_streaming
MultiscaleFilterBanks.ReadOnlyTaps
```
