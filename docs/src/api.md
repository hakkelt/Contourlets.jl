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
ct_forward
ct_forward!
ct_inverse
ct_inverse!
nsct_forward
nsct_forward!
nsct_inverse
nsct_inverse!
tict_forward
tict_forward!
tict_inverse
tict_inverse!
stict_forward
stict_forward!
stict_inverse
stict_inverse!
```

## Types

```@docs
ContourletParams
ContourletCoefficients
NSCTCoefficients
TICTCoefficients
CTWorkspace
NSCTWorkspace
TICTWorkspace
STICTWorkspace
```


## Workspace

```@docs
make_workspace
make_nsct_workspace
similar_coefficients
similar_nsct_coefficients
make_tict_workspace
make_stict_workspace
similar_tict_coefficients
similar_stict_coefficients
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

The filter utilities below are public but not exported.

```@docs
upsample_kernel
check_pr_condition
```
