"""
    Contourlets

Two-dimensional contourlet transforms, as transform objects of
[MultiscaleTransformsBase](https://github.com/hakkelt/MultiscaleTransformsBase.jl),
whose interface this package re-exports.

## Quick reference

| Name | Description |
|------|-------------|
| [`CT`](@ref) | Discrete contourlet transform (nearly critically sampled) |
| [`NSCT`](@ref) | Nonsubsampled contourlet transform (shift-invariant) |
| [`TICT`](@ref) | Translation-invariant contourlet transform (Eslami & Radha) |
| [`STICT`](@ref) | Semi-translation-invariant contourlet transform (TI pyramid, decimated DFB) |
| `analyze` / `synthesize` (and `!` forms) | Run a transform |
| `make_workspace` | Filters, buffers and FFT plans for repeated calls on one size |
| `subband_layout` / `nsubbands` / `subband` | What each subband is, and where it is in the coefficients |
| [`parabolic_levels`](@ref) | Per-scale DFB depths for parabolic scaling |

```julia
using Contourlets
t = NSCT(J = 3, L_array = parabolic_levels(3))
c = analyze(t, x)          # size (size(x)..., nsubbands(t, size(x)))
x̂ = synthesize(t, c)
```

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
[MultiscaleFilterBanks.jl](https://github.com/hakkelt/MultiscaleFilterBanks.jl);
its filter types `FilterPair` and `CDF97` are re-exported here.

### Building blocks
The pyramid and directional stages (`lp_*`, `nsp_*`, `qfb_*`, `dfb_*`,
`nsdfb_*`) and the filter utilities `upsample_kernel` and `check_pr_condition`
are public but not exported: load them by name, e.g.
`using Contourlets: lp_decompose, dfb_decompose`.
"""
module Contourlets

using FFTW: FFTW, plan_rfft, plan_fft, plan_fft!, plan_irfft, plan_brfft, plan_bfft, plan_bfft!
using LinearAlgebra: mul!
using FastBroadcast: @..
using Compat: @compat

using MultiscaleFilterBanks: MultiscaleFilterBanks,
    FilterPair, CDF97,
    data_eltype, filter_eltype, with_fftw_threads, clamp_idx, nsp_atrous_filters, nsp_responses, centered_dtft,
    MaskItem, MaskBank, MaskBankEngine, mask_bank_forward!, mask_bank_inverse!, mask_bank_tune

# The interface's functions get their methods by qualified definitions
# (`MultiscaleTransformsBase.analyze!(...) = ...`).
using MultiscaleTransformsBase: MultiscaleTransformsBase, AbstractMultiscaleTransform, AbstractTransformWorkspace,
    SubbandVector, SubbandLayout, check_workspace,
    analyze, analyze!, synthesize, synthesize!, make_workspace, transform,
    similar_coefficients, subband_layout, nsubbands, subband

# ── Types (must come before filters which use them) ──────────────────────────
include("types/quincunx_filter_pair.jl")
include("types/transforms.jl")

# ── Filter coefficients and utilities ────────────────────────────────────────
include("filters/q2345.jl")
include("filters/filter_utils.jl")

# ── DSP primitives ────────────────────────────────────────────────────────────
include("primitives/threading.jl")
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
include("workspace/mask_bank.jl")   # stacked NSCT/TICT coefficients for the shared engine
include("workspace/contourlet_workspace.jl")

# ── Top-level transforms ──────────────────────────────────────────────────────
include("transforms/ct.jl")
include("transforms/nsct.jl")
include("transforms/tilp_transforms.jl")
include("transforms/tict.jl")
include("transforms/stict.jl")
include("transforms/common.jl")

# ── Precompilation ────────────────────────────────────────────────────────────
include("precompile.jl")

# ── Exports ───────────────────────────────────────────────────────────────────
# The transform API is exported.  The pyramid and directional stages and the
# filter utilities are public but not exported; the raw primitives (conv2d*,
# rect_*sample*, qx_*sample*, shear*, inv_shear*) are internal.
export
    # Transforms and their workspaces
    CT, NSCT, TICT, STICT,
    CTWorkspace, NSCTWorkspace, TICTWorkspace, STICTWorkspace,

    # Filters (FilterPair and CDF97 re-exported from MultiscaleFilterBanks)
    FilterPair, CDF97, QuincunxFilterPair, Q2345,

    # Utilities
    parabolic_levels,

    # The interface (MultiscaleTransformsBase)
    AbstractMultiscaleTransform, AbstractTransformWorkspace,
    analyze, analyze!, synthesize, synthesize!, make_workspace, transform, similar_coefficients,
    subband_layout, nsubbands, subband, SubbandVector, SubbandLayout

# Building blocks
@compat public lp_decompose, lp_decompose!, lp_reconstruct, lp_reconstruct!,
    nsp_decompose, nsp_decompose!, nsp_reconstruct, nsp_reconstruct!,
    qfb_decompose, qfb_decompose!, qfb_reconstruct, qfb_reconstruct!,
    dfb_decompose, dfb_decompose!, dfb_reconstruct, dfb_reconstruct!, dfb_subband_sizes,
    nsdfb_decompose, nsdfb_reconstruct,
    upsample_kernel, check_pr_condition

end # module
