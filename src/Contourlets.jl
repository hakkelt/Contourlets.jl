"""
    Contourlets

Discrete Contourlet Transform (CT) and Nonsubsampled Contourlet Transform (NSCT)
implemented as a standalone Julia package.

## Quick reference

| Function | Description |
|----------|-------------|
| `ct_forward` / `ct_inverse` | Discrete CT (nearly critically sampled) |
| `nsct_forward` / `nsct_inverse` | NSCT (fully shift-invariant) |
| `ct_forward!` / `ct_inverse!` | In-place CT using a preallocated workspace |
| `nsct_forward!` / `nsct_inverse!` | In-place NSCT |
| `make_workspace` | Allocate a reusable `ContourletWorkspace` |
| `parabolic_levels` | Compute optimal per-scale DFB depth array |

### Default filters
- `CDF97` — CDF 9/7 biorthogonal filter pair for the Laplacian Pyramid stage.
- `Q2345` — Quincunx biorthogonal filter pair for the DFB stage.
"""
module Contourlets

using FFTW
using LinearAlgebra

# ── Types (must come before filters which use them) ──────────────────────────
include("types.jl")

# ── Filter coefficients and utilities ────────────────────────────────────────
include("filters/cdf97.jl")
include("filters/q2345.jl")
include("filters/filter_utils.jl")

# ── DSP primitives ────────────────────────────────────────────────────────────
include("primitives/conv2d.jl")
include("primitives/sampling.jl")
include("primitives/quincunx.jl")
include("primitives/shearing.jl")

# ── Pyramid stages ────────────────────────────────────────────────────────────
include("pyramid/laplacian_pyramid.jl")
include("pyramid/nonsubsampled_pyramid.jl")

# ── Directional filter bank ───────────────────────────────────────────────────
include("directional/quincunx_fb.jl")
include("directional/dfb.jl")
include("directional/nsdfb.jl")

# ── Workspace ─────────────────────────────────────────────────────────────────
include("workspace.jl")

# ── Top-level transforms ──────────────────────────────────────────────────────
include("transforms/ct.jl")
include("transforms/nsct.jl")

# ── Precompilation ────────────────────────────────────────────────────────────
include("precompile.jl")

# ── Exports ───────────────────────────────────────────────────────────────────
export
    # Types
    FilterPair, QuincunxFilterPair,
    ContourletParams, ContourletCoefficients, NSCTCoefficients,
    ContourletWorkspace,

    # Default filters
    CDF97, Q2345,

    # Utilities
    parabolic_levels,
    upsample_filter, upsample_kernel, check_pr_condition,

    # Primitives
    conv2d!, conv2d, conv2d_sep!, conv2d_sep,
    rect_downsample!, rect_downsample, rect_upsample!, rect_upsample,
    qx_downsample!, qx_downsample, qx_upsample!, qx_upsample,
    shear!, shear, inv_shear!, inv_shear,

    # Building blocks
    lp_decompose, lp_decompose!, lp_reconstruct, lp_reconstruct!,
    nsp_decompose, nsp_decompose!, nsp_reconstruct, nsp_reconstruct!,
    qfb_decompose, qfb_decompose!, qfb_reconstruct, qfb_reconstruct!,
    dfb_decompose, dfb_reconstruct, dfb_subband_sizes,
    nsdfb_decompose, nsdfb_reconstruct,

    # Top-level transforms
    ct_forward, ct_forward!, ct_inverse, ct_inverse!,
    nsct_forward, nsct_forward!, nsct_inverse, nsct_inverse!,
    similar_coefficients, similar_nsct_coefficients,

    # Workspace
    make_workspace, make_nsct_workspace,
    estimate_workspace_size, workspace_clear!

end # module
