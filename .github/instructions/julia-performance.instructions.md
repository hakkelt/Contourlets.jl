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

## FFTW Plans

Do **not** add a package-level plan cache.  FFTW already caches its `MEASURE`
wisdom (the expensive timing step) globally in the C library, so calling
`plan_rfft(buf; flags=FFTW.MEASURE)` again for a size seen before is cheap
(sub-millisecond).  A Julia-side `Dict` of plan objects only saves that
sub-millisecond reconstruction and adds a lock plus a type-unstable cache.

Prefer `FFTW.MEASURE` for repeated transforms (the wisdom amortizes the cost),
or `FFTW.ESTIMATE` when a plan is used only once and the first-call latency of
`MEASURE` matters.

## Workspace Pattern

Hot paths in iterative algorithms must accept a `ContourletWorkspace` and
write into its pre-allocated buffers.  Do not allocate inside `ct_forward!` /
`ct_inverse!` / `nsct_forward!` / `nsct_inverse!`.

## Benchmark Discipline

The persistent suite in `benchmark/benchmarks.jl` uses `@benchmarkable`
(deferred — `run(SUITE)` executes it later):

```julia
# Good — uses pre-allocated data, deterministic seed
rng  = MersenneTwister(1234)
img  = randn(rng, 256, 256)
SUITE["CT"]["256"]["forward"] = @benchmarkable ct_forward($img, $params)

# Bad — allocates inside the benchmark, non-deterministic
SUITE["CT"]["256"]["forward"] = @benchmarkable ct_forward(randn(256, 256), params)
```

For a quick one-off measurement at the REPL, use `@benchmark` (runs immediately
and prints a full distribution) or `@btime` (prints just the minimum time and
allocations):

```julia
using BenchmarkTools
img = randn(MersenneTwister(1234), 256, 256)
p   = ContourletParams(J=2, L_array=[2, 3])

@benchmark ct_forward($img, $p)     # full timing distribution
@btime ct_forward($img, $p);        # one-line min time + allocations
```

Always `$`-interpolate captured variables (`$img`, `$p`) in `@benchmark`,
`@btime`, and `@benchmarkable` so the value is not treated as a non-constant
global — otherwise the measurement is dominated by global-lookup overhead.
