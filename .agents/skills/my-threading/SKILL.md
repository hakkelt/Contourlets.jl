---
name: my-threading
description: Work with the dual-path threading architecture (LoopVectorization @turbo for real data, Polyester @batch for complex data). Use when adding threaded loops, modifying ThreadingPolicy dispatch, or diagnosing threading-related performance issues.
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
# All top-level transforms accept a threading kwarg:
ct_forward(image, params; threading=Auto())     # default
ct_forward(image, params; threading=Enabled())  # force multi-core
ct_forward(image, params; threading=Disabled()) # force single-thread

# Same for ct_inverse!, nsct_forward!, etc.
```

`ThreadingPolicy` types are exported: `Auto`, `Enabled`, `Disabled`.

## Reference Files

- **[Policy dispatch](references/policy.md)** — `_use_threading`, how `Auto` resolves per type
- **[Writing threaded loops](references/loops.md)** — `@turbo` vs `@batch`, placement, guards
- **[Adding threading to a new function](references/adding.md)** — step-by-step checklist

## Quick Check

```julia
using Contourlets
# Auto selects correctly:
Contourlets._use_threading(Auto(), Float64)    # → false (SIMD path)
Contourlets._use_threading(Auto(), ComplexF64) # → true  (batch path)
```

## Key Invariants

- Never put `@turbo` on a loop that operates on complex arrays — it will error or silently produce wrong results.
- Never spawn `Base.Threads.@spawn` tasks inside hot recursive paths — task creation overhead dominates.
- `@batch` (Polyester) reuses a persistent thread pool; it is safe to call recursively and in tight loops.
- `Disabled()` must suppress both `@turbo` and `@batch` — fall back to a plain `for` loop.

## Related Skills

- `julia-perf` — diagnosing whether threading is helping or hurting (use `@btime` with `threading=Disabled()` vs `Auto()`)
- `my-gpu-ext` — GPU transforms ignore `ThreadingPolicy` (kernels run on device threads)
