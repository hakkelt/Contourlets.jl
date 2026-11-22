# [API Reference](@id api)

```@meta
CurrentModule = Contourlets
```

## Module

```@docs
Contourlets
MultiscaleFilterBanks.MultiscaleFilterBanks
```

## Transforms

```@docs
CT
NSCT
TICT
STICT
```

## Running a transform

The verbs and accessors are those of
[MultiscaleTransformsBase](https://github.com/hakkelt/MultiscaleTransformsBase.jl),
re-exported here; each entry below lists the interface's contract first and the
contourlet methods after it.  Every verb also takes a workspace in place of the
transform: `analyze(ws, x)`, `analyze!(c, ws, x)`, `synthesize(ws, c)` and
`synthesize!(x, ws, c)`.

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

## From MultiscaleTransformsBase

The interface's abstract types, the public helpers an implementation uses, and
its conformance tests.

```@docs
MultiscaleTransformsBase.MultiscaleTransformsBase
AbstractMultiscaleTransform
AbstractTransformWorkspace
MultiscaleTransformsBase.DefaultWorkspace
MultiscaleTransformsBase.check_workspace
MultiscaleTransformsBase.signal_size
MultiscaleTransformsBase.supports_streaming
MultiscaleTransformsBase.Testing
MultiscaleTransformsBase.Testing.test_transform
```

## Building blocks

The pyramid and directional stages below are public but not exported; load them
by name, e.g. `using Contourlets: lp_decompose, dfb_decompose`.

### Pyramid

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

### Directional Filter Bank

```@docs
qfb_decompose
qfb_decompose!
qfb_reconstruct
qfb_reconstruct!
dfb_decompose
dfb_decompose!
dfb_reconstruct
dfb_reconstruct!
dfb_subband_sizes
nsdfb_decompose
nsdfb_reconstruct
```

## Filters

```@docs
CDF97
Q2345
FilterPair
QuincunxFilterPair
parabolic_levels
```

`FilterPair` and `CDF97` come from MultiscaleFilterBanks, as do these two
helpers their documentation refers to:

```@docs
MultiscaleFilterBanks.upsample_filter
MultiscaleFilterBanks.ReadOnlyTaps
```

The filter utility below is public but not exported.

```@docs
upsample_kernel
```
