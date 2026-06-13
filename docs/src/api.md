# API Reference

```@meta
CurrentModule = Contourlets
```

## Module

```@docs
Contourlets
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
```

## Types

```@docs
ContourletParams
ContourletCoefficients
NSCTCoefficients
ContourletWorkspace
```

## Workspace

```@docs
make_workspace
make_nsct_workspace
similar_coefficients
similar_nsct_coefficients
workspace_clear!
estimate_workspace_size
```

## Pyramid

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

## Directional Filter Bank

```@docs
qfb_decompose
qfb_decompose!
qfb_reconstruct
qfb_reconstruct!
dfb_decompose
dfb_reconstruct
dfb_subband_sizes
nsdfb_decompose
nsdfb_reconstruct
```

## Primitives

```@docs
conv2d
conv2d!
conv2d_sep
conv2d_sep!
rect_downsample
rect_downsample!
rect_upsample
rect_upsample!
qx_downsample
qx_downsample!
qx_upsample
qx_upsample!
shear
shear!
inv_shear
inv_shear!
```

## Filters

```@docs
CDF97
Q2345
FilterPair
QuincunxFilterPair
parabolic_levels
upsample_filter
upsample_kernel
check_pr_condition
```
