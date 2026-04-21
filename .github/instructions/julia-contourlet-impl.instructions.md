---
applyTo: "src/**/*.jl"
---

# Implementation Guidelines — Contourlets.jl

## Filter Storage Conventions

- `FilterPair{T}` stores `h` (analysis) and `g` (synthesis) as `Vector{T}`.
- `QuincunxFilterPair{T}` stores `h_q`, `g_q` as `Matrix{T}` (1×N by convention).
- All filter constants are defined once in `src/filters/` and referenced by
  value — never recreate filter arrays inside hot loops.
- When converting precision: use `T.(fp.h)` rather than `convert.(T, fp.h)`.

## In-Place (`!`) Patterns

- Every public function `f(args...)` must have an in-place companion
  `f!(dst, args...)` that writes into `dst` without allocating.
- The allocating wrapper is simply `f(args...) = f!(similar(first_output(args...)), args...)`.
- Accept a `workspace::Union{ContourletWorkspace, Nothing} = nothing` kwarg on
  allocating variants so iterative algorithms can opt in to reuse.

## Memory Layout

- Julia arrays are **column-major**. Inner loops must iterate over rows (`i`)
  while outer loops iterate over columns (`j`):
  ```julia
  @inbounds for j in 1:n2, i in 1:n1
      dst[i, j] = ...
  end
  ```
- Never use `size(A, 1)` inside a hot inner loop — extract to a local variable.

## Boundary Conditions

- Default: `:symmetric` (mirror extension). Also support `:periodic`.
- Pass boundary mode as `Val(:symmetric)` / `Val(:periodic)` to allow
  compiler dispatch rather than runtime branching.

## Shearing Index Convention

- Horizontal shear: `(i, j) → (i, mod1(j + i, n2))`
- Vertical shear: `(i, j) → (mod1(i + j, n1), j)`
- Inverse is the same operation with opposite sign in the modular step.
- Shearing is a pure index remap — no arithmetic on array values.

## Code Style

- Use `@inbounds` inside `@inbounds` blocks only where bounds have been
  manually verified.
- Avoid scalar loops over complex expressions; prefer fused broadcast:
  `@. dst = src1 - src2`
- Runic formatting is mandatory. Run `runic src/` before committing.
- No type annotations wider than needed — prefer duck typing for public API.
