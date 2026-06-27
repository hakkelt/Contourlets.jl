# Data Movement: Host ↔ Device

## Uploading Data to the Device

```julia
backend = _gpu_backend(image)          # detect backend from an existing device array

# Move a CPU array to the device (no-op if already there)
x_d = _to_device(backend, x_cpu)      # explicit copy
h_d = _ensure_gpu(backend, h)         # for filters: Vector or Matrix → device Vector/Matrix
```

`_to_device` always allocates and copies. `_ensure_gpu` is a no-op for arrays already on the device.

## Downloading Coefficients to the Host

```julia
# Single array
x_cpu = Array(x_d)

# Entire coefficient set (ContourletCoefficients or NSCTCoefficients)
using KernelAbstractions: Adapt
coeffs_cpu = Adapt.adapt(Array, coeffs_gpu)

# Upload a CPU coefficient set to a specific device
coeffs_gpu = Adapt.adapt(CuArray, coeffs_cpu)   # CUDA example
```

`Adapt.adapt_structure` is defined for both `ContourletCoefficients` and `NSCTCoefficients` in `transforms_gpu.jl` — it recursively adapts all nested arrays.

## Type Conversions

```julia
Td = Contourlets._data_eltype(image)      # data type (real or complex)
Tf = Contourlets._filter_eltype(Td)       # = real(float(Td))

# Convert a CPU filter to the right precision before uploading
h_d = _ensure_gpu(backend, Tf.(fp.h))
```

Always convert before uploading: `Tf.(fp.h)` produces a CPU `Vector{Tf}`, then `_ensure_gpu` uploads it once.

## Workspace and Workspaces on GPU

A `ContourletWorkspace` built from a GPU image is GPU-resident:

```julia
ws = make_workspace(cu_image, params)   # scratch buffers are CuMatrix
ct_forward!(coeffs, cu_image, params; workspace=ws)
```

CUDA workspaces trigger graph capture automatically in `ContourletsCUDAExt` (see `cuda-graphs.md`). For other backends, the workspace path still avoids per-call allocations but does not capture.
