# [Reuse a workspace](@id guide-workspaces)

Each call of [`analyze`](@ref) or [`synthesize`](@ref) without a workspace
builds the filters, buffers and FFT plans it needs and then discards them. A
*workspace*, built once by [`make_workspace`](@ref), keeps them for repeated
calls on images of one size and element type, as in iterative reconstruction or
in a parameter sweep.

## Build a workspace and pass it to every call

```@example workspace
using Contourlets, Random

img = randn(Xoshiro(1), 256, 256)
t = NSCT(scales = 3, direction_levels = [2, 3, 3])

ws = make_workspace(t, img)              # for images like img
c = similar_coefficients(t, img)         # coefficients for analyze!
out = similar(img)

for _ in 1:10
    analyze!(c, t, img; workspace = ws)
    # … modify the coefficients c …
    synthesize!(out, t, c; workspace = ws)
end
maximum(abs, out .- img)
```

`make_workspace(t, img)` takes the size, the element type and the array type
from `img`, so a GPU array gives a GPU workspace;
`make_workspace(t, Float32, (512, 512))` takes them explicitly. The `!` forms
write into an output you allocate, here with [`similar_coefficients`](@ref) and
`similar`. A workspace also works with the allocating forms,
`analyze(t, img; workspace = ws)`, and knows its transform, so
`analyze!(c, ws, img)` and `synthesize!(out, ws, c)` need no `t`.

A call with a workspace for another transform, element type or image size
throws an error. A workspace serves one call at a time: tasks that run
transforms concurrently need one workspace each.

## What it saves

The time per analysis with and without a workspace, measured while these pages
were built:

```@example workspace
using BenchmarkTools

t_alloc = @belapsed analyze($t, $img)
t_ws = @belapsed analyze!($c, $t, $img; workspace = $ws)
println("without workspace: ", round(t_alloc * 1e3; digits = 2), " ms")
println("with workspace:    ", round(t_ws * 1e3; digits = 2), " ms")
```

## Construction options

[`make_workspace`](@ref) takes these keywords:

- `flags`: the FFTW planner flags, `FFTW.MEASURE` by default.
  `flags = FFTW.ESTIMATE` builds the workspace faster, possibly with slower
  plans; use it when a workspace serves only a few calls.
- `tasks`: the largest number of tasks an NSCT or TICT workspace splits its
  work between (see [Use threads and GPUs](@ref guide-parallel)).
- `tune`: for NSCT and TICT workspaces on a GPU, measure the fastest batch
  sizes at construction.
- `prewarm`: run one analysis and synthesis at construction (the default), so
  that the first call does not pay for compilation.

`Base.summarysize(ws)` gives the memory a workspace holds.
