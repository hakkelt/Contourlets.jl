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

Both directional filter banks filter on the DFT grid with the equivalent filters
of the DFB tree, which are evaluated in closed form for the image size at hand.
That evaluation is the part a workspace saves:

- **Allocating** (`nsct_forward(x, params)`, `ct_forward(x, params)`) builds the
  equivalent filters, FFT buffers and plans on every call.  Convenient for
  one-shot use.
- **Workspace** ([`make_nsct_workspace`](@ref) + `nsct_forward!`,
  [`make_workspace`](@ref) + `ct_forward!`) builds them once.  Each later call is
  an FFT, pointwise multiplies (plus an alias fold for the decimated DFB) and
  inverse FFTs, and allocates nothing.  An NSCT workspace also holds the
  nonsubsampled pyramid as masks on the input's spectrum, so `nsct_forward!` is
  one FFT of the image and one inverse FFT per output array, with no
  convolution at all.

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

The NSCT and TICT transforms take a `threading` keyword (the CT and the STICT
run their stages one after another and take none).  It is a [`ThreadingPolicy`](@ref) —
`Auto()` (the default), `Enabled()` or `Disabled()` — and the default is a hybrid:

| Data | Default | Why |
|---|---|---|
| Real (`Float32`/`Float64`) | single-threaded `LoopVectorization.@turbo` | the inner loops vectorise cleanly; SIMD beats thread-spawn overhead |
| Complex | multi-threaded `Polyester.@batch` | the loops do not vectorise; `@batch` spawns at near-zero cost |

`Enabled()` and `Disabled()` override that choice.  The NSCT and TICT workspace
paths are FFT-bound instead: there, threading runs the independent subband FFTs
in one task per Julia thread (FFTW's own threads, which split a single FFT, gain
little at image sizes).  Real data stays single-threaded by default there too;
pass `threading = Enabled()` to use the cores:

```@example perf
p_t = ContourletParams(J = 3, L_array = [2, 3, 4])
ws_t = make_tict_workspace(p_t, (256, 256))
buf_t = similar_tict_coefficients(p_t, (256, 256))
t_serial = @belapsed tict_forward!($buf_t, $img, $p_t; workspace = $ws_t, threading = Disabled())
t_threaded = @belapsed tict_forward!($buf_t, $img, $p_t; workspace = $ws_t, threading = Enabled())
println("tict_forward! serial   : $(round(t_serial * 1e3; digits = 1)) ms")
println("tict_forward! threaded : $(round(t_threaded * 1e3; digits = 1)) ms on $(Threads.nthreads()) thread(s)")
```

## GPU

`ContourletsGPUExt` loads when a KernelAbstractions backend (CUDA, AMDGPU, Metal,
oneAPI) is present and runs *every* stage of the CT and NSCT on the device;
coefficients stay device-resident.  The directional stage needs an FFT for the
array type (cuFFT with CUDA, rocFFT with AMDGPU); the pyramid and the shear-based
primitives run on any backend.  Pass a device array and dispatch does the rest:

```julia
using CUDA, Contourlets
coeffs = ct_forward(CuArray(img), params)   # ContourletCoefficients of CuArrays
img_back = Array(ct_inverse(coeffs, params))
```

`ContourletsCUDAExt` additionally captures the workspace `ct_forward!` /
`ct_inverse!` paths into CUDA graphs, replacing their many small kernel launches
per call with a single replay. Without graph capture — on AMDGPU —
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
