"""
    Contourlets

Discrete Contourlet Transform (CT) and Nonsubsampled Contourlet Transform (NSCT)
implemented as a standalone Julia package.

## Quick reference

| Function | Description |
|----------|-------------|
| `ct_forward` / `ct_inverse` | Discrete CT (nearly critically sampled) |
| `nsct_forward` / `nsct_inverse` | NSCT (fully shift-invariant) |
| `nsct3_forward` / `nsct3_inverse` | 3-D NSCT on volumes, with a streaming form |
| `ct_forward!` / `ct_inverse!` | In-place CT using a preallocated workspace |
| `nsct_forward!` / `nsct_inverse!` | In-place NSCT |
| `make_workspace` | Allocate a reusable `ContourletWorkspace` |
| `parabolic_levels` | Compute optimal per-scale DFB depth array |
| `make_nsct3_workspace` | Allocate a reusable `NSCT3Workspace` for volumes |

### Default filters
- `CDF97` — CDF 9/7 biorthogonal filter pair for the Laplacian Pyramid stage.
- `Q2345` — Quincunx biorthogonal filter pair for the DFB stage.

### 3-D
The 3-D transform is a parallel API rather than a generalisation of the 2-D one:
the directional stage shares nothing with the quincunx filter bank, and is built
instead from Bernstein mapping kernels (`make_hourglass3`, `make_nsdfb3`).  See
the "Theory (3-D)" page.
"""
module Contourlets

using FFTW: FFTW, plan_rfft, plan_fft, plan_irfft, plan_ifft, rfft, fft
using LinearAlgebra: mul!, adjoint
using LoopVectorization: @turbo
using Polyester: @batch

# ── Types (must come before filters which use them) ──────────────────────────
include("threading.jl")
include("types.jl")
include("types3.jl")

# ── Filter coefficients and utilities ────────────────────────────────────────
include("filters/cdf97.jl")
include("filters/q2345.jl")
include("filters/filter_utils.jl")
include("filters/bernstein.jl")

# ── DSP primitives ────────────────────────────────────────────────────────────
include("primitives/conv2d.jl")
include("primitives/conv3d_fft.jl")
include("primitives/conv3d.jl")
include("primitives/sampling.jl")
include("primitives/quincunx.jl")
include("primitives/shearing.jl")

# ── Pyramid stages ────────────────────────────────────────────────────────────
include("pyramid/laplacian_pyramid.jl")
include("pyramid/nonsubsampled_pyramid.jl")
include("pyramid/nonsubsampled_pyramid3.jl")

# ── Directional filter bank ───────────────────────────────────────────────────
include("directional/quincunx_fb.jl")
include("directional/dfb.jl")
include("directional/nsdfb.jl")
include("directional/hourglass3.jl")
include("directional/nsdfb3.jl")

# ── Workspace ─────────────────────────────────────────────────────────────────
include("workspace.jl")

# ── Top-level transforms ──────────────────────────────────────────────────────
include("transforms/ct.jl")
include("transforms/nsct.jl")
include("transforms/nsct3.jl")

# ── Precompilation ────────────────────────────────────────────────────────────
include("precompile.jl")

# ── Exports ───────────────────────────────────────────────────────────────────
export
    # Types
    FilterPair, QuincunxFilterPair,
    ContourletParams, ContourletCoefficients, NSCTCoefficients,
    ContourletParams3, NSCT3Coefficients, NSCT3Workspace,
    ContourletWorkspace,

    # Default filters
    CDF97, Q2345,

    # Utilities
    parabolic_levels,
    upsample_filter, upsample_kernel, check_pr_condition,

    # Primitives
    conv2d!, conv2d, conv2d_sep!, conv2d_sep,
    conv3d_sep!, conv3d_sep, conv3d_sep_fft!, conv3d_sep_fft,
    Conv3DFFTPlan, make_conv3d_fft_plan,
    rect_downsample!, rect_downsample, rect_upsample!, rect_upsample,
    qx_downsample!, qx_downsample, qx_upsample!, qx_upsample,
    shear!, shear, inv_shear!, inv_shear,

    # Building blocks
    lp_decompose, lp_decompose!, lp_reconstruct, lp_reconstruct!,
    nsp_decompose, nsp_decompose!, nsp_reconstruct, nsp_reconstruct!,
    nsp3_decompose, nsp3_decompose!, nsp3_reconstruct, nsp3_reconstruct!,
    qfb_decompose, qfb_decompose!, qfb_reconstruct, qfb_reconstruct!,
    dfb_decompose, dfb_decompose!, dfb_reconstruct, dfb_reconstruct!, dfb_subband_sizes,
    nsdfb_decompose, nsdfb_reconstruct,
    HourglassFB3, make_hourglass3, hourglass3_mask,
    hourglass3_decompose, hourglass3_decompose!,
    hourglass3_reconstruct, hourglass3_reconstruct!,
    NSDFB3, make_nsdfb3, nsdfb3_sharing, nsdfb3_nsubbands, nsdfb3_subband_index, nsdfb3_subband_triple, nsdfb3_mask,
    nsdfb3_decompose, nsdfb3_decompose!,
    nsdfb3_reconstruct, nsdfb3_reconstruct!,

    # Top-level transforms
    ct_forward, ct_forward!, ct_inverse, ct_inverse!,
    nsct_forward, nsct_forward!, nsct_inverse, nsct_inverse!,
    nsct3_forward, nsct3_inverse, nsct3_inverse!, nsct3_nsubbands, make_nsct3_workspace,
    similar_coefficients, similar_nsct_coefficients,

    # Workspace
    make_workspace, make_nsct_workspace,
    estimate_workspace_size,

    ThreadingPolicy, Auto, Enabled, Disabled

end # module
