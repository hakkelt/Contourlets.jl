"""
    Contourlets

Discrete Contourlet Transform (CT) and Nonsubsampled Contourlet Transform (NSCT)
implemented as a standalone Julia package.

## Quick reference

| Function | Description |
|----------|-------------|
| `ct_forward` / `ct_inverse` | Discrete CT (nearly critically sampled) |
| `nsct_forward` / `nsct_inverse` | NSCT (fully shift-invariant) |
| `nsct3d_forward` / `nsct3d_inverse` | 3-D NSCT on volumes, with a streaming form |
| `ct_forward!` / `ct_inverse!` | In-place CT using a preallocated workspace |
| `nsct_forward!` / `nsct_inverse!` | In-place NSCT |
| `make_workspace` | Allocate a reusable `ContourletWorkspace` |
| `parabolic_levels` | Compute optimal per-scale DFB depth array |
| `make_nsct3d_workspace` | Allocate a reusable `NSCT3DWorkspace` for volumes |

### Default filters
- `CDF97` — CDF 9/7 biorthogonal filter pair for the Laplacian Pyramid stage.
- `Q2345` — Quincunx biorthogonal filter pair for the DFB stage.

### 3-D
The 3-D transform is a parallel API rather than a generalisation of the 2-D one:
the directional stage shares nothing with the quincunx filter bank, and is built
instead from Bernstein mapping kernels (`make_hourglass3d`, `make_nsdfb3d`).  See
the "Theory (3-D)" page.
"""
module Contourlets

using FFTW: FFTW, plan_rfft, plan_fft, plan_fft!, plan_irfft, plan_ifft, plan_ifft!, rfft, fft
using LinearAlgebra: mul!, adjoint
using LoopVectorization: @turbo
using Polyester: @batch

# ── Types (must come before filters which use them) ──────────────────────────
include("threading.jl")
include("types.jl")
include("types3d.jl")

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
include("pyramid/nonsubsampled_pyramid3d.jl")

# ── Directional filter bank ───────────────────────────────────────────────────
include("directional/quincunx_fb.jl")
include("directional/dfb.jl")
include("directional/nsdfb.jl")
include("directional/hourglass3d.jl")
include("directional/nsdfb3d.jl")

# ── Workspace ─────────────────────────────────────────────────────────────────
include("workspace.jl")

# ── Top-level transforms ──────────────────────────────────────────────────────
include("transforms/ct.jl")
include("transforms/nsct.jl")
include("transforms/nsct3d.jl")
include("transforms/surfacelet3d.jl")

# ── Precompilation ────────────────────────────────────────────────────────────
include("precompile.jl")

# ── Exports ───────────────────────────────────────────────────────────────────
export
    # Types
    FilterPair, QuincunxFilterPair,
    ContourletParams, ContourletCoefficients, NSCTCoefficients,
    ContourletParams3D, NSCT3DCoefficients, NSCT3DWorkspace,
    SurfaceletParams3D, Surfacelet3DCoefficients, Surfacelet3DPlan,
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
    nsp3d_decompose, nsp3d_decompose!, nsp3d_reconstruct, nsp3d_reconstruct!,
    qfb_decompose, qfb_decompose!, qfb_reconstruct, qfb_reconstruct!,
    dfb_decompose, dfb_decompose!, dfb_reconstruct, dfb_reconstruct!, dfb_subband_sizes,
    nsdfb_decompose, nsdfb_reconstruct,
    HourglassFB3D, make_hourglass3d, hourglass3d_mask, hourglass3d_weight,
    hourglass3d_decompose, hourglass3d_decompose!,
    hourglass3d_reconstruct, hourglass3d_reconstruct!,
    NSDFB3D, make_nsdfb3d, nsdfb3d_sharing, nsdfb3d_nsubbands, nsdfb3d_subband_index, nsdfb3d_subband_triple, nsdfb3d_mask, nsdfb3d_weight,
    nsdfb3d_decompose, nsdfb3d_decompose!,
    nsdfb3d_reconstruct, nsdfb3d_reconstruct!,

    # Top-level transforms
    ct_forward, ct_forward!, ct_inverse, ct_inverse!,
    nsct_forward, nsct_forward!, nsct_inverse, nsct_inverse!,
    nsct3d_forward, nsct3d_inverse, nsct3d_inverse!, nsct3d_nsubbands, make_nsct3d_workspace,
    surfacelet3d_forward, surfacelet3d_inverse, surfacelet3d_nsubbands,
    surfacelet3d_redundancy, make_surfacelet3d,
    similar_coefficients, similar_nsct_coefficients,

    # Workspace
    make_workspace, make_nsct_workspace,
    estimate_workspace_size,

    ThreadingPolicy, Auto, Enabled, Disabled

end # module
