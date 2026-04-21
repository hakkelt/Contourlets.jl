# API Reference

```@meta
CurrentModule = Contourlets
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
nsp_reconstruct
```

## Directional Filter Bank

```@docs
dfb_decompose
dfb_reconstruct
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
upsample_filter
check_pr_condition
```
