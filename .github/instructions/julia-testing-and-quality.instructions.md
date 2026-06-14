---
applyTo: "test/**/*.jl"
---

# Testing Guidelines — Contourlets.jl

## Framework

All tests use `@testitem` blocks from TestItemRunner.  Each block is
self-contained: it must `using Contourlets` (and any other deps) inside the
block.

## Tag Taxonomy

| Tag | Description |
|-----|-------------|
| `:filters` | Filter coefficient correctness, PR conditions |
| `:primitives` | Sampling, shearing, convolution |
| `:pyramid` | LP and NSP decompose/reconstruct |
| `:directional` | DFB and NSDFB |
| `:ct` | Full Contourlet Transform |
| `:nsct` | Nonsubsampled CT (includes `:ct` too) |
| `:quality` | Aqua, JET |

Example: `@testitem "LP PR" tags=[:pyramid]`

## PR Test Tolerances

| Transform | Tolerance |
|-----------|-----------|
| LP one level | `< 1e-14` |
| NSP one level | `< 1e-14` |
| DFB (any L) | `< 1e-13` |
| NSDFB | `< 1e-13` |
| CT (J=2) | `< 1e-12` |
| NSCT (J=2) | `< 2e-15` |
| Float32 | `< 1e-4` |

## Shift-Invariance Test Pattern

```julia
@testitem "NSCT shift-invariance" tags=[:nsct] begin
    using Contourlets, Random
    Random.seed!(42)
    x = randn(64, 64)
    p = ContourletParams(J=2, L_array=[2, 3])
    shift = (3, 7)
    ns   = nsct_forward(x, p)
    ns_s = nsct_forward(circshift(x, shift), p)
    for j in 1:p.J, k in eachindex(ns.subbands[j])
        @test maximum(abs, ns_s.subbands[j][k] .- circshift(ns.subbands[j][k], shift)) < 1e-10
    end
end
```

## Determinism

Always seed random inputs: `Random.seed!(42)` (or another fixed seed).  Never
rely on the default global RNG state.
