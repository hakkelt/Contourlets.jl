---
name: my-threading
description: Work with the dual-path threading architecture (LoopVectorization @turbo for real data, Polyester @batch for complex data). Use when adding threaded loops, modifying the `threaded` keyword resolution, or diagnosing threading-related performance issues.
---

# Threading Architecture

Contourlets.jl uses a hybrid approach to achieve near-identical wall-clock
performance for both `Real` and `Complex` data:

| Data type | Backend | Mechanism |
|---|---|---|
| `Real` (`Float32`/`Float64`) | `LoopVectorization.@turbo` | SIMD on a single thread |
| `Complex` (`ComplexF32`/`ComplexF64`) | `Polyester.@batch` | multi-core via persistent thread pool |

`LoopVectorization` does not support complex numbers, so the two paths are
mutually exclusive. `Polyester`'s near-zero spawn overhead makes multi-core
`ComplexF64` transforms competitive with single-thread SIMD `Float64` transforms.

## Public API

```julia
# The transforms that thread (NSCT, TICT, STICT) accept a `threaded` keyword:
nsct_forward(image, params; threaded = nothing)     # default
nsct_forward(image, params; threaded = true)  # force multi-core
nsct_forward(image, params; threaded = false) # force single-thread

# Same for nsct_inverse!, tict_forward!, etc.  The CT takes no `threaded` keyword.
```

`threaded::Union{Bool, Nothing}`: `nothing` lets the transform decide.

## Reference Files

- **[Policy dispatch](references/policy.md)** — `use_threading`, how `threaded = nothing` resolves per type
- **[Writing threaded loops](references/loops.md)** — `@turbo` vs `@batch`, placement, guards
- **[Adding threading to a new function](references/adding.md)** — step-by-step checklist

## Quick Check

```julia
using Contourlets
# the default selects correctly:
Contourlets.use_threading(nothing, Float64)    # → false (SIMD path)
Contourlets.use_threading(nothing, ComplexF64) # → true  (batch path)
```

## Key Invariants

- Never put `@turbo` on a loop that operates on complex arrays — it will error or silently produce wrong results.
- Never spawn `Base.Threads.@spawn` tasks inside hot recursive paths — task creation overhead dominates.
- `@batch` (Polyester) reuses a persistent thread pool; it is safe to call recursively and in tight loops.
- `threaded = false` must suppress both `@turbo` and `@batch` — fall back to a plain `for` loop.

## Related Skills

- `julia-perf` — diagnosing whether threading is helping or hurting (use `@btime` with `threaded = false` vs the default)
- `my-gpu-ext` — GPU transforms ignore `threaded` (kernels run on device threads)
