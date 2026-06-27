# Writing GPU Kernels

## KernelAbstractions Pattern

```julia
using KernelAbstractions

@kernel function my_kernel!(dst, @Const(src), n::Int)
    i, j = @index(Global, NTuple)
    @inbounds dst[i, j] = src[i, j] * 2
end

function my_op!(dst::_AbstractGPUMatrix{T}, src::_AbstractGPUMatrix{T}) where {T}
    n1, n2 = size(src)
    backend = _gpu_backend(src)
    kernel = my_kernel!(backend, (16, 16))          # 16×16 workgroup for 2-D
    kernel(dst, src, n1; ndrange = (n1, n2))
    return dst
end
```

- **2-D kernels**: `(16, 16)` workgroup.
- **1-D kernels**: `256` workgroup; use scalar `@index(Global)`.
- `@Const` on read-only arrays avoids aliasing overhead.

## Boundary Handling in Kernels

Use the module-local `_gpu_boundary_idx(p, n, bmode)` helper:
- `bmode = 1` → symmetric reflect
- `bmode = 2` → periodic (mod-based)

```julia
ii = _gpu_boundary_idx(i - (k - center), n1, bmode)
```

This helper uses uniform branching (no warp divergence for typical filter sizes).

## Complex Data

GPU kernels work with the data type `T` (which may be complex). Keep filters as `real(T)`:

```julia
Tf = real(T)
h_d = _ensure_gpu(backend, Tf.(h))    # real filter on device
# inside kernel: acc += h[k] * src[ii, jj]  — works for T complex or real
```

## Allocating Device Buffers

```julia
# Prefer:
dst = KernelAbstractions.allocate(backend, T, m, n)   # uninitialized
dst = KernelAbstractions.zeros(backend, T, m, n)       # zero-initialized (needed for += accumulation)
dst = _scratch_like(src, m, n)                         # same type/backend as src, uninitialized

# Never:
dst = zeros(T, m, n)    # always CPU!
dst = Matrix{T}(undef, m, n)  # always CPU!
```

Use `KernelAbstractions.zeros` only when the kernel uses `+=` accumulation (e.g. synthesis/reconstruction); otherwise `allocate` is faster.

## No Index-Vector Gathers

Index-vector gathers (`A[[1,3,5], :]`) cause scalar fallback on most backends. Use `circshift` / `circshift!` for cyclic rolls:

```julia
# Good
y = circshift!(similar(x), x, (0, -1))   # roll columns left by 1

# Bad — forces scalar indexing
y = x[:, [2:end; 1]]
```
