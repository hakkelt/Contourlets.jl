# GPU Primitive Overloads

## Dispatch Pattern

All GPU primitives are overloaded for `_AbstractGPUMatrix`, defined in `ext/ContourletsGPUExt.jl`:

```julia
const GPUSubArray{T, N} = SubArray{T, N, <:AbstractGPUArray}
const _AbstractGPUMatrix{T} = Union{AbstractGPUMatrix{T}, GPUSubArray{T, 2}}
```

CPU methods use `AbstractMatrix`; GPU overloads use `_AbstractGPUMatrix`. Julia dispatch picks the more specific GPU method when given device arrays.

## Existing Overloads (ext/ContourletsGPUExt/primitives_gpu.jl)

| Primitive | CPU signature | GPU kernel |
|---|---|---|
| `conv2d_sep!` | `dst, src, h_row, h_col` | `_conv_cols_kernel!` + `_conv_rows_kernel!` |
| `shear!` / `inv_shear!` | `dst, src, dir` | `_shear_h/v_kernel!` |
| `rect_downsample!` | `dst, src` | `_rect_downsample_kernel!` |
| `rect_upsample!` | `dst, src` | `_rect_upsample_kernel!` |
| `qx_downsample!` / `qx_upsample!` | `dst, src` | `_qx_downsample/upsample_kernel!` |
| `qfb_decompose` / `qfb_reconstruct` | `image, qfp; dir` | `_qfb_{col,row}_{decompose,reconstruct}_kernel_gpu!` |

## Adding a New Primitive Overload

1. Write the `@kernel` function in `primitives_gpu.jl`.
2. Write the dispatch function overloading the CPU method for `_AbstractGPUMatrix`.
3. Add the symbol to the `import Contourlets: ...` list in `ContourletsGPUExt.jl`.
4. Add a test in `test/items/test_gpu.jl` with tag `:gpu`.

## `conv2d_sep!` Filter Cache

`conv2d_sep!` on GPU checks `Contourlets._active_filter_cache()` before uploading filters. During CUDA graph capture, the CUDA extension pre-builds device filter vectors and stores them in the cache so `_ensure_gpu` finds them instead of allocating graph memory nodes. When writing new ops that upload filters in a hot loop, use the same pattern:

```julia
fcache = Contourlets._active_filter_cache()
h_d = if fcache !== nothing && haskey(fcache, h)
    fcache[h]
else
    _ensure_gpu(backend, Tf.(h))
end
```
