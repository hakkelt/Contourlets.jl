# Contourlets.jl

*Discrete Contourlet Transform and Nonsubsampled Contourlet Transform in Julia.*

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/hakkelt/Contourlets.jl")
```

## Quick Start

```julia
using Contourlets, Random

Random.seed!(42)
img = randn(256, 256)

# Choose scale/direction parameters (parabolic scaling rule)
params = ContourletParams(J=4, L_array=parabolic_levels(4))

# ── Contourlet Transform (critically sampled) ────────────────────────────────
coeffs = ct_forward(img, params)
rec    = ct_inverse(coeffs, params)
@assert maximum(abs, rec .- img) < 1e-12

# ── Nonsubsampled CT (shift-invariant) ───────────────────────────────────────
ns_coeffs = nsct_forward(img, params)
ns_rec    = nsct_inverse(ns_coeffs, params)
@assert maximum(abs, ns_rec .- img) < 1e-12

# ── Preallocated-buffer iterative usage ──────────────────────────────────────
ws  = make_workspace(Float64, size(img), params)
buf = similar_coefficients(params, size(img))
for _ in 1:100
    ct_forward!(buf, img, params; workspace=ws)   # reuses workspace buffers across iterations
end
```

### 3-D volumes

```julia
vol     = randn(Float32, 128, 128, 128)
params3 = ContourletParams3(J=2, L_array=[2, 2])   # 12 directional subbands per scale
ws3     = make_nsct3_workspace(params3, size(vol), Float32)

c3   = nsct3_forward(vol, params3; workspace=ws3)
rec3 = nsct3_inverse(c3, params3; workspace=ws3)
@assert maximum(abs, rec3 .- vol) < 1f-4

# Streaming: the form to use at working sizes, where the full coefficient set
# does not fit — 256³ Float32 with J=3, L=[2,2,3] is 3.1 GB.
coarse = nsct3_forward(vol, params3; workspace=ws3) do j, k, band
    # `band` is valid only until this callback returns
end
```

## Package Overview

| Component | Description |
|-----------|-------------|
| [`ContourletParams`](@ref) | Bundles scale `J`, direction counts `L_array`, and filter pair |
| [`ct_forward`](@ref) / [`ct_inverse`](@ref) | Discrete Contourlet Transform |
| [`nsct_forward`](@ref) / [`nsct_inverse`](@ref) | Nonsubsampled Contourlet Transform |
| [`make_workspace`](@ref) | Preallocate buffers for iterative algorithms |
| [`parabolic_levels`](@ref) | Compute optimal direction counts per scale |
| [`CDF97`](@ref) | Default CDF 9/7 biorthogonal LP filter pair |
| [`Q2345`](@ref) | Default "23-45" (Phoong et al. 1995) ladder DFB filter pair |
| [`ContourletParams3`](@ref) | 3-D counterpart: scales, per-scale `L`, Bernstein order |
| [`nsct3_forward`](@ref) / [`nsct3_inverse`](@ref) | 3-D NSCT, with streaming forms |
| [`make_nsct3_workspace`](@ref) | Preallocate 3-D plans and buffers |
| [`make_nsdfb3`](@ref) | 3-D directional filter bank on its own (`3·2^L` subbands) |

See the [Theory](@ref theory) page and [Theory — 3-D](@ref theory3d) for the mathematical background, and the [API Reference](@ref) for the complete public interface.

## Performance & Threading

Contourlets.jl provides an explicit `ThreadingPolicy` configuration via `threading` keyword arguments. By default, it uses the `Auto()` policy which delivers maximum throughput via a hybrid architecture:
- **Real-valued Data:** Multithreading is disabled by default, utilizing `LoopVectorization.@turbo` for zero-overhead, single-threaded SIMD acceleration.
- **Complex-valued Data:** Multithreading is enabled by default, utilizing `Polyester.@batch` to distribute non-vectorizable inner loops efficiently across CPU cores with near-zero latency.
Users can explicitly opt into or out of multithreading by passing `threading=Enabled()` or `threading=Disabled()`.
- **GPU Acceleration:** `ContourletsGPUExt` seamlessly runs the full transform (pyramid, directional, and sampling stages) on device-resident arrays for any GPU backend supported by KernelAbstractions.jl (CUDA, AMDGPU, Metal, oneAPI). Pass a device array and the correct operations dispatch automatically. Note that the **CUDA backend** (`ContourletsCUDAExt`) additionally captures the workspace `ct_forward!`/`ct_inverse!` paths into CUDA graphs, reducing per-call host overhead from ~300 kernel launches to a single replay. Without this optimization — i.e. on AMDGPU, Metal, or oneAPI — GPU execution is still correct but only outperforms CPU for sufficiently large images where device compute dominates over host-side launch overhead.

