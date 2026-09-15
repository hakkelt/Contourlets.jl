# [Performance, threading and GPUs](@id performance)

Absolute timings depend on the machine, so every number on this page is measured
while the documentation is built and is reported together with the machine that
produced it:

```@example perf
println("Julia      : ", VERSION)
println("CPU        : ", Sys.CPU_NAME, ", ", Sys.CPU_THREADS, " logical cores")
println("Threads    : ", Threads.nthreads(), " Julia thread(s)")
println("OS         : ", Sys.KERNEL, " ", Sys.MACHINE)
```

Documentation builds run on a GitHub-hosted runner, which is a shared virtual
machine — treat the ratios as meaningful and the absolute numbers as indicative.
Cross-implementation timings against the MATLAB toolboxes, measured on dedicated
hardware, are in [Comparison with the MATLAB toolboxes](@ref comparison).

## Workspaces are the main lever

The NSCT has two execution paths with very different profiles:

- **Allocating** (`nsct_forward(x, params)`) runs the 2-D fan/parallelogram
  convolutions directly. Convenient for one-shot use; the kernels skip the
  structural zeros of the à trous filters (~70% of each) and use SIMD or
  multi-threaded batching.
- **Workspace** ([`make_nsct_workspace`](@ref) + `nsct_forward!`) precomputes the
  `2^L` equivalent leaf filters once in the frequency domain. Each later call is
  an FFT, a pointwise multiply and an inverse FFT.

```@example perf
using Contourlets, BenchmarkTools

params = ContourletParams(J = 2, L_array = [2, 3])
img    = randn(256, 256)
ws     = make_nsct_workspace(params, (256, 256))
buf    = similar_nsct_coefficients(params, (256, 256))

t_alloc = @belapsed nsct_forward($img, $params)
t_ws    = @belapsed nsct_forward!($buf, $img, $params; workspace = $ws)

println("nsct_forward  allocating : $(round(t_alloc * 1e3; digits = 1)) ms")
println("nsct_forward! workspace  : $(round(t_ws    * 1e3; digits = 1)) ms")
println("speedup                  : $(round(t_alloc / t_ws; digits = 1))×")
```

For anything that transforms the same size repeatedly — gain estimation, noise
calibration, iterative reconstruction — the workspace form is the right choice.

The critically sampled CT uses the decimated DFB rather than the NSDFB, so it is
substantially cheaper at equal image size:

```@example perf
t_ct   = @belapsed ct_forward($img, $params)
t_nsct = @belapsed nsct_forward($img, $params)

println("ct_forward   (256×256, J=2, L=[2,3]): $(round(t_ct   * 1e3; digits = 1)) ms")
println("nsct_forward                        : $(round(t_nsct * 1e3; digits = 1)) ms")
println("ratio                               : $(round(t_nsct / t_ct; digits = 1))×")
```

That is the cost of shift invariance: the NSCT keeps every subband at full
resolution, so its redundancy is `1 + Σ_j 2^{L_j}` rather than ≈ 4/3.

## Threading

The `threading` keyword takes a [`ThreadingPolicy`](@ref) — `Auto()` (the
default), `Enabled()` or `Disabled()` — and the default is a hybrid:

| Data | Default | Why |
|---|---|---|
| Real (`Float32`/`Float64`) | single-threaded `LoopVectorization.@turbo` | the inner loops vectorise cleanly; SIMD beats thread-spawn overhead |
| Complex | multi-threaded `Polyester.@batch` | the loops do not vectorise; `@batch` spawns at near-zero cost |

`Enabled()` and `Disabled()` override that choice. Note that FFT threading in an
NSCT workspace is fixed when the workspace is constructed — see
[2-D transforms](@ref guide2d).

## GPU

`ContourletsGPUExt` loads when a KernelAbstractions backend (CUDA, AMDGPU, Metal,
oneAPI) is present and runs *every* stage of the CT and NSCT on the device;
coefficients stay device-resident. Pass a device array and dispatch does the
rest:

```julia
using CUDA, Contourlets
coeffs = ct_forward(CuArray(img), params)   # ContourletCoefficients of CuArrays
img_back = Array(ct_inverse(coeffs, params))
```

`ContourletsCUDAExt` additionally captures the workspace `ct_forward!` /
`ct_inverse!` paths into CUDA graphs, replacing roughly 300 kernel launches per
call with a single replay. Without graph capture — on AMDGPU, Metal or oneAPI —
the device path is still correct, but only wins once the image is large enough
for compute to dominate host-side launch overhead. There is a fixed per-call
floor of a few milliseconds on the 2-D transforms, so small images are faster on
the CPU.

## Reproducing

```bash
julia --project=benchmark --startup-file=no -e '
    include("benchmark/benchmarks.jl"); using BenchmarkTools
    BenchmarkTools.save(".temp/benchmark_results.json", run(SUITE; verbose=true))'
```

`benchmark/gpu_benchmark.jl` covers the device paths. To compare two revisions:

```bash
benchpkg Contourlets --path . --rev=master,HEAD --script=benchmark/benchmarks.jl \
    --output-dir=.temp/bench --exeflags="--threads=2"
benchpkgtable Contourlets --path . --rev=master,HEAD --input-dir=.temp/bench --ratio
```
