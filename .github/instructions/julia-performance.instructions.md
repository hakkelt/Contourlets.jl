---
applyTo: "src/**/*.jl,benchmark/**/*.jl"
---

# Performance Guidelines — Contourlets.jl

## Type Stability

Every public function must be type-stable.  Verify with:
```julia
using JET
JET.@report_opt ct_forward(randn(32, 32), ContourletParams(J=2, L_array=[2,3]))
```

Common pitfalls:
- Indexing into a `Vector{Any}` subbands structure — always parametrise on `T`.
- `promote_type` returning a wider type than needed — prefer `float(eltype(x))`.
- `similar(x)` inside a parametric function — ensure the return type is inferred.

## `@inbounds` Usage

Only annotate `@inbounds` where:
1. Loop bounds are derived directly from `size(array)`, OR
2. You have manually verified no out-of-bounds access is possible.

Never annotate `@inbounds` on code that indexes into user-supplied arrays
without bounds checking.

## Column-Major Inner Loops

Julia stores arrays in column-major order.  Cache-friendly pattern:
```julia
@inbounds for j in 1:n2   # outer: columns
    for i in 1:n1          # inner: rows (contiguous in memory)
        dst[i, j] = src[i, j] + 1
    end
end
```

## FFTW Plan Reuse

FFTW plans are expensive to create.  Cache plans indexed by `(src_size, kernel_size)`:
```julia
const _PLAN_CACHE = Dict{Tuple{NTuple{2,Int},NTuple{2,Int}}, Any}()
```

Use `FFTW.MEASURE` for cached plans (one-time overhead), `FFTW.ESTIMATE` in
contexts where the plan is used only once.

## Workspace Pattern

Hot paths in iterative algorithms must accept a `ContourletWorkspace` and
write into its pre-allocated buffers.  Do not allocate inside `ct_forward!` /
`ct_inverse!` / `nsct_forward!` / `nsct_inverse!`.

## Benchmark Discipline

```julia
# Good — uses pre-allocated data, deterministic seed
rng  = MersenneTwister(1234)
img  = randn(rng, 256, 256)
SUITE["CT"]["256"]["forward"] = @benchmarkable ct_forward($img, $params)

# Bad — allocates inside the benchmark, non-deterministic
SUITE["CT"]["256"]["forward"] = @benchmarkable ct_forward(randn(256, 256), params)
```

Always use `$` interpolation for captured variables in `@benchmarkable`.
