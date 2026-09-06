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

## Transforms (3-D)

```@docs
nsct3_forward
nsct3_inverse
nsct3_inverse!
nsct3_nsubbands
make_nsct3_workspace
```

## Surfacelet transform (3-D, decimated)

```@docs
SurfaceletParams3
Surfacelet3Coefficients
Surfacelet3Plan
make_surfacelet3
surfacelet3_forward
surfacelet3_inverse
surfacelet3_nsubbands
surfacelet3_redundancy
```

## Types

```@docs
ContourletParams
ContourletCoefficients
NSCTCoefficients
ContourletWorkspace
ContourletParams3
NSCT3Coefficients
NSCT3Workspace
```


## Threading

```@docs
ThreadingPolicy
Auto
Enabled
Disabled
```

## Workspace

```@docs
make_workspace
make_nsct_workspace
similar_coefficients
similar_nsct_coefficients
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
nsp3_decompose
nsp3_decompose!
nsp3_reconstruct
nsp3_reconstruct!
```

## Directional Filter Bank

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

## Directional Filter Bank (3-D)

```@docs
HourglassFB3
make_hourglass3
hourglass3_mask
hourglass3_weight
hourglass3_decompose
hourglass3_decompose!
hourglass3_reconstruct
hourglass3_reconstruct!
NSDFB3
make_nsdfb3
nsdfb3_sharing
nsdfb3_nsubbands
nsdfb3_subband_index
nsdfb3_subband_triple
nsdfb3_mask
nsdfb3_weight
nsdfb3_decompose
nsdfb3_decompose!
nsdfb3_reconstruct
nsdfb3_reconstruct!
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
conv3d_sep
conv3d_sep!
conv3d_sep_fft
conv3d_sep_fft!
Conv3DFFTPlan
make_conv3d_fft_plan
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

## Filter design internals (3-D)

Not exported — these are the pieces the 3-D hourglass bank is built from, useful
if you want to design your own mapping kernels.

```@docs
bernstein_table
dft_frequencies
mapping_coefficients
mapping_response
halfband_split
```
