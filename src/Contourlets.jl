"""
    Contourlets

Discrete Contourlet Transform (CT) and Nonsubsampled Contourlet Transform (NSCT)
implemented as a standalone Julia package.

## Quick reference

| Function | Description |
|----------|-------------|
| `ct_forward` / `ct_inverse` | Discrete CT (nearly critically sampled) |
| `nsct_forward` / `nsct_inverse` | NSCT (fully shift-invariant) |
| `tict_forward` / `tict_inverse` | Translation-invariant CT (Eslami & Radha) |
| `stict_forward` / `stict_inverse` | Semi-translation-invariant CT (TI pyramid, decimated DFB) |
| `ct_forward!` / `ct_inverse!` | In-place CT using a preallocated workspace |
| `nsct_forward!` / `nsct_inverse!` | In-place NSCT |
| `make_workspace` / `make_nsct_workspace` | Allocate a reusable `CTWorkspace` / `NSCTWorkspace` |
| `parabolic_levels` | Compute optimal per-scale DFB depth array |

### Default filters
- `CDF97` — CDF 9/7 biorthogonal filter pair for the Laplacian Pyramid stage.
- `Q2345` — Quincunx biorthogonal filter pair for the DFB stage.

### Volumes
This package is planar throughout: the quincunx lattice, the shear index remap
and the Q2345 ladder all rely on properties of the plane.  Volumes are covered
by [Surfacelets.jl](https://github.com/hakkelt/Surfacelets.jl), which is a
different construction rather than this one with an extra axis — its directional
stage is an hourglass filter bank built from Bernstein mapping kernels, with no
sampling matrix or ladder filter in it at all.  The two packages share only the
pyramid filter machinery, which lives in
[MultiscaleFilterBanks.jl](https://github.com/hakkelt/MultiscaleFilterBanks.jl)
and is re-exported here.
"""
module Contourlets

using FFTW: FFTW, plan_rfft, plan_fft, plan_fft!, plan_irfft, plan_ifft, plan_bfft!, rfft, irfft, fft, ifft
using LinearAlgebra: mul!, adjoint
using LoopVectorization: @turbo
using FastBroadcast: @..

using MultiscaleFilterBanks: MultiscaleFilterBanks,
    FilterPair, CDF97, upsample_filter,
    ThreadingPolicy, Auto, Enabled, Disabled,
    _data_eltype, _filter_eltype, _use_threading, _n_tasks, _foreach_chunk, _chunked_sum!,
    _loop_threaded, _with_fftw_threads, _clamp_idx, _nsp_atrous_filters, _nsp_responses, _centred_dtft

# ── Types (must come before filters which use them) ──────────────────────────
include("types/quincunx_filter_pair.jl")
include("types/params.jl")
include("types/coefficients.jl")

# ── Filter coefficients and utilities ────────────────────────────────────────
include("filters/q2345.jl")
include("filters/filter_utils.jl")

# ── DSP primitives ────────────────────────────────────────────────────────────
include("primitives/conv2d.jl")
include("primitives/conv2d_sep.jl")
include("primitives/sampling.jl")
include("primitives/quincunx.jl")
include("primitives/shearing.jl")

# ── Pyramid stages ────────────────────────────────────────────────────────────
include("pyramid/laplacian_pyramid.jl")
include("pyramid/nonsubsampled_pyramid.jl")
include("pyramid/tilp.jl")

# ── Directional filter bank ───────────────────────────────────────────────────
include("directional/quincunx_fb.jl")
include("directional/bank_design.jl")
include("directional/bank.jl")
include("directional/bank_runner.jl")
include("directional/dfb.jl")
include("directional/nsdfb.jl")

# ── Workspace ─────────────────────────────────────────────────────────────────
include("workspace/scratch_arena.jl")
include("workspace/mask_bank.jl")
include("workspace/contourlet_workspace.jl")

# ── Top-level transforms ──────────────────────────────────────────────────────
include("transforms/ct.jl")
include("transforms/nsct.jl")
include("transforms/tilp_transforms.jl")
include("transforms/tict.jl")
include("transforms/stict.jl")

# ── Precompilation ────────────────────────────────────────────────────────────
include("precompile.jl")

# ── Exports ───────────────────────────────────────────────────────────────────
export
    # Re-exported from MultiscaleFilterBanks
    FilterPair, CDF97, upsample_filter,
    ThreadingPolicy, Auto, Enabled, Disabled,

    # Types
    QuincunxFilterPair,
    ContourletParams, ContourletCoefficients, NSCTCoefficients, TICTCoefficients,
    CTWorkspace, NSCTWorkspace, TICTWorkspace, STICTWorkspace,

    # Default filters
    Q2345,

    # Utilities
    parabolic_levels,
    upsample_kernel, check_pr_condition,

    # Primitives
    conv2d!, conv2d, conv2d_sep!, conv2d_sep,
    rect_downsample!, rect_downsample, rect_upsample!, rect_upsample,
    qx_downsample!, qx_downsample, qx_upsample!, qx_upsample,
    shear!, shear, inv_shear!, inv_shear,

    # Building blocks
    lp_decompose, lp_decompose!, lp_reconstruct, lp_reconstruct!,
    nsp_decompose, nsp_decompose!, nsp_reconstruct, nsp_reconstruct!,
    qfb_decompose, qfb_decompose!, qfb_reconstruct, qfb_reconstruct!,
    dfb_decompose, dfb_decompose!, dfb_reconstruct, dfb_reconstruct!, dfb_subband_sizes,
    nsdfb_decompose, nsdfb_reconstruct,

    # Top-level transforms
    ct_forward, ct_forward!, ct_inverse, ct_inverse!,
    nsct_forward, nsct_forward!, nsct_inverse, nsct_inverse!,
    tict_forward, tict_forward!, tict_inverse, tict_inverse!,
    stict_forward, stict_forward!, stict_inverse, stict_inverse!,
    similar_coefficients, similar_nsct_coefficients,
    similar_tict_coefficients, similar_stict_coefficients,

    # Workspace
    make_workspace, make_nsct_workspace, make_tict_workspace, make_stict_workspace

end # module
