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
nsct3d_forward
nsct3d_inverse
nsct3d_inverse!
nsct3d_nsubbands
make_nsct3d_workspace
```

## Surfacelet transform (3-D, decimated)

```@docs
SurfaceletParams3D
Surfacelet3DCoefficients
Surfacelet3DPlan
make_surfacelet3d
surfacelet3d_forward
surfacelet3d_inverse
surfacelet3d_nsubbands
surfacelet3d_redundancy
```

## Types

```@docs
ContourletParams
ContourletCoefficients
NSCTCoefficients
ContourletWorkspace
ContourletParams3D
NSCT3DCoefficients
NSCT3DWorkspace
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
nsp3d_decompose
nsp3d_decompose!
nsp3d_reconstruct
nsp3d_reconstruct!
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
HourglassFB3D
make_hourglass3d
hourglass3d_mask
hourglass3d_weight
hourglass3d_decompose
hourglass3d_decompose!
hourglass3d_reconstruct
hourglass3d_reconstruct!
NSDFB3D
make_nsdfb3d
nsdfb3d_sharing
nsdfb3d_nsubbands
nsdfb3d_subband_index
nsdfb3d_subband_triple
nsdfb3d_mask
nsdfb3d_weight
nsdfb3d_decompose
nsdfb3d_decompose!
nsdfb3d_reconstruct
nsdfb3d_reconstruct!
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
