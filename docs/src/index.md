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
    ct_forward!(buf, img, ws)   # reuses workspace buffers across iterations
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

See the [Theory](@ref theory) page for the mathematical background, and the [API Reference](@ref) for the complete public interface.

## Performance & Threading

Contourlets.jl provides an explicit `ThreadingPolicy` configuration via `threading` keyword arguments. By default, it uses the `Auto()` policy which delivers maximum throughput via a hybrid architecture:
- **Real-valued Data:** Multithreading is disabled by default, utilizing `LoopVectorization.@turbo` for zero-overhead, single-threaded SIMD acceleration.
- **Complex-valued Data:** Multithreading is enabled by default, utilizing `Polyester.@batch` to distribute non-vectorizable inner loops efficiently across CPU cores with near-zero latency.
Users can explicitly opt into or out of multithreading by passing `threading=Enabled()` or `threading=Disabled()`.
- **GPU Acceleration:** `ContourletsGPUExt` seamlessly runs the transforms on device-resident arrays for any supported GPU array backend via KernelAbstractions.jl.

