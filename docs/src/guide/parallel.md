# [Use threads and GPUs](@id guide-parallel)

## Threads

The [`NSCT`](@ref) and the [`TICT`](@ref) can split the FFTs of their subbands
between tasks on several threads. Start Julia with threads, for example
`julia --threads=4`, and pass the `threaded` keyword:

- `threaded = true` splits the work into one task per Julia thread, or into at
  most the `tasks` given to [`make_workspace`](@ref);
- `threaded = false` runs on the calling task;
- `threaded = nothing`, the default, threads complex data and not real data.

This applies to the NSCT when it runs with a workspace (see
[Reuse a workspace](@ref guide-workspaces)) and to the TICT always. The
[`CT`](@ref) and the [`STICT`](@ref) run serially and ignore `threaded`.

```@example parallel
using Contourlets, BenchmarkTools, Random

img = randn(Xoshiro(1), 256, 256)
t = TICT(scales = 3, direction_levels = [2, 3, 4])
ws = make_workspace(t, img)
c = similar_coefficients(t, img)

t_serial = @belapsed analyze!($c, $t, $img; workspace = $ws, threaded = false)
t_threaded = @belapsed analyze!($c, $t, $img; workspace = $ws, threaded = true)
println("serial:   ", round(t_serial * 1e3; digits = 1), " ms")
println("threaded: ", round(t_threaded * 1e3; digits = 1), " ms on ",
    Threads.nthreads(), " threads")
```

To run several transforms in parallel yourself, give each task its own
workspace and build it with `tasks = 1`, so that the workspace's calls stay on
that task.

## GPUs

The transforms run on GPU arrays: arrays of GPUArrays.jl with a
KernelAbstractions.jl backend whose array type has an FFT, as those of CUDA.jl
and AMDGPU.jl have. Pass a GPU image; the coefficients and workspaces stay on
the device:

```julia
using CUDA, Contourlets

img = CuArray(rand(Float32, 256, 256))
t = NSCT(scales = 3, direction_levels = [2, 3, 3])
ws = make_workspace(t, img)                  # a GPU workspace
c = analyze(t, img; workspace = ws)          # a CuArray
img_back = synthesize(t, c; workspace = ws)
```

OpenCL.jl arrays have no FFT. Metal.jl supports `Float32` only. For NSCT and TICT
workspaces on a GPU, `make_workspace(t, img; tune = true)` measures the fastest
batch sizes once at construction, which pays off for a workspace that serves
many calls.
