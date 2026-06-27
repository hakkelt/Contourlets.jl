---
name: my-gpu-ext
description: Work on the ContourletsGPUExt / ContourletsCUDAExt package extensions. Use when writing GPU kernels, adding GPU primitives, porting CPU code to the device, or debugging GPU correctness.
---

# GPU Extension (`ContourletsGPUExt`)

This skill covers the GPU extension architecture specific to Contourlets.jl.
For general Julia GPU programming, refer to KernelAbstractions.jl docs.

## Architecture Overview

- **Whole-transform on device** — every stage (pyramid, DFB/NSDFB) runs on the device; nothing is implicitly transferred back to the host.
- **Pyramid reuse** — `lp_decompose`, `nsp_decompose`, etc. are *not* overloaded for GPU. They use broadcasts that dispatch to the overloaded GPU primitives (`conv2d_sep!`, `rect_downsample!`, …) automatically when given device arrays.
- **Coefficients are device-resident** — `ContourletCoefficients{Td,A}` and `NSCTCoefficients{Td,A}` carry a generic `A<:AbstractMatrix` storage type. A GPU forward pass returns coefficients whose arrays are still on the device.
- **CUDA graph capture** — `ContourletsCUDAExt` wraps the workspace (`_ct_forward_ws!` / `_ct_inverse_ws!`) paths in CUDA graph capture for maximum replay throughput. See `references/cuda-graphs.md`.

## Key Invariants (never break these)

- **Scratch allocation**: use `_scratch_like(A, m, n)` or `similar(A, ...)`, never bare `zeros(T, m, n)` — that lands on the CPU.
- **Filter upload**: `_ensure_gpu(backend, filter)` moves a CPU Vector/Matrix to the device; no-op if already there. Use `_to_device(backend, x)` for explicit copies.
- **Filters stay real**: GPU kernels keep filters as `real(T)` and accumulate in the data type `T` — mirrors the CPU split (`Td`/`Tf`).
- **No index-vector gathers**: use `circshift` / `circshift!` for rolls — index gathers force scalar indexing on most GPU backends.
- **Size from array, not `Matrix`**: `KernelAbstractions.allocate(backend, T, m, n)` not `Matrix{T}(undef, m, n)`.
- **Backend detection**: `_gpu_backend(A)` → `KernelAbstractions.get_backend(A)`.

## Reference Files

- **[Kernels](references/kernels.md)** — writing `@kernel` functions, workgroup sizes, index patterns
- **[Primitives](references/primitives.md)** — overloading conv2d_sep!, shear!, sampling ops for `_AbstractGPUMatrix`
- **[Data Movement](references/data-movement.md)** — `_to_device`, `_ensure_gpu`, `Adapt.adapt`, filter caches
- **[CUDA Graphs](references/cuda-graphs.md)** — how `ContourletsCUDAExt` captures and replays the workspace path
- **[Testing](references/testing.md)** — JLArrays (CI), real backends, `:gpu` tag, GPUEnv.jl

## Quick Diagnostic

```julia
# Confirm GPU round-trip matches CPU (to Float32 precision)
using Contourlets, JLArrays
x = jl(rand(Float32, 64, 64))
params = ContourletParams(J=2)
coeffs = ct_forward(x, params)
x_rec  = ct_inverse(coeffs, params)
@assert maximum(abs, Array(x_rec) .- Array(x)) < 1f-4

# Move coefficients to host
coeffs_cpu = Adapt.adapt(Array, coeffs)
```

## Related Skills

- `julia-perf` — CPU performance patterns (allocations, type stability)
- `julia-bench` — benchmarking GPU vs CPU paths
