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

- **Allocating** (`analyze(t, x)`, `synthesize(t, c)`) builds the
  equivalent filters, FFT buffers and plans on every call.  Convenient for
  one-shot use.
- **Workspace** ([`make_workspace`](@ref) + `analyze!(c, t, x; workspace = ws)`)
  builds them once.  Each later call is
  an FFT, pointwise multiplies (plus an alias fold for the decimated DFB) and
  inverse FFTs, and allocates nothing.  An NSCT workspace also holds the
  nonsubsampled pyramid as masks on the input's spectrum, so its `analyze!` is
  one FFT of the image and one inverse FFT per output array, with no
  convolution at all.

```@example perf
using Contourlets, BenchmarkTools

tn     = NSCT(J = 2, L_array = [2, 3])
img    = randn(256, 256)
ws     = make_workspace(tn, img)
buf    = similar_coefficients(tn, img)

t_alloc = @belapsed analyze($tn, $img)
t_ws    = @belapsed analyze!($buf, $tn, $img; workspace = $ws)

println("NSCT analyze  allocating : $(round(t_alloc * 1e3; digits = 1)) ms")
println("NSCT analyze! workspace  : $(round(t_ws    * 1e3; digits = 1)) ms")
println("speedup                  : $(round(t_alloc / t_ws; digits = 1))×")
```

For anything that transforms the same size repeatedly — gain estimation, noise
calibration, iterative reconstruction — the workspace form is the right choice.

The critically sampled CT uses the decimated DFB rather than the NSDFB, so it is
substantially cheaper at equal image size:

```@example perf
tc     = CT(J = 2, L_array = [2, 3])
t_ct   = @belapsed analyze($tc, $img)
t_nsct = @belapsed analyze($tn, $img)

println("CT analyze   (256×256, J=2, L=[2,3]): $(round(t_ct   * 1e3; digits = 1)) ms")
println("NSCT analyze                        : $(round(t_nsct * 1e3; digits = 1)) ms")
println("ratio                               : $(round(t_nsct / t_ct; digits = 1))×")
```

That is the cost of shift invariance: the NSCT keeps every subband at full
resolution, so its redundancy is `1 + Σ_j 2^{L_j}` rather than ≈ 4/3.

## Threading

The NSCT and TICT transforms take a `threaded` keyword (the CT and the STICT
run their stages one after another; they accept it and ignore it): `true`, `false`, or `nothing`
(the default), which lets the transform decide.  The default is a hybrid:

| Data | Default | Why |
|---|---|---|
| Real (`Float32`/`Float64`) | single-threaded `@simd` loops | the inner loops vectorise cleanly; SIMD beats thread-spawn overhead |
| Complex | multi-threaded `Polyester.@batch` | the loops do not vectorise; `@batch` spawns at near-zero cost |

`threaded = true` and `threaded = false` override that choice.  The NSCT and TICT workspace
paths are FFT-bound instead: there, threading runs the independent subband FFTs
in one task per Julia thread (FFTW's own threads, which split a single FFT, gain
little at image sizes).  Real data stays single-threaded by default there too;
pass `threaded = true` to use the cores:

```@example perf
tt = TICT(J = 3, L_array = [2, 3, 4])
ws_t = make_workspace(tt, img)
buf_t = similar_coefficients(tt, img)
t_serial = @belapsed analyze!($buf_t, $tt, $img; workspace = $ws_t, threaded = false)
t_threaded = @belapsed analyze!($buf_t, $tt, $img; workspace = $ws_t, threaded = true)
println("TICT analyze! serial   : $(round(t_serial * 1e3; digits = 1)) ms")
println("TICT analyze! threaded : $(round(t_threaded * 1e3; digits = 1)) ms on $(Threads.nthreads()) thread(s)")
```

## GPU

`ContourletsGPUExt` loads when a KernelAbstractions backend is present and runs
*every* stage of the transforms on the device; coefficients stay device-resident.
The directional stage needs an FFT for the array type (cuFFT with CUDA, rocFFT
with AMDGPU, Metal's FFT for `Float32`); the pyramid and the shear-based
primitives run on any backend.  OpenCL.jl has no FFT: with `CLArray`s in shared
virtual memory (OpenCL.jl's default on devices that support it) FFTW transforms
them on the CPU, so the results are right but the transforms run about twice
as slowly as on the CPU, and buffer-backed `CLArray`s fail.  Pass a device array and
dispatch does the rest:

```julia
using CUDA, Contourlets
t = CT(J = 2, L_array = [2, 3])
coeffs = analyze(t, CuArray(img))        # a SubbandVector over a CuArray
img_back = Array(synthesize(t, coeffs))
```

`ContourletsCUDAExt` additionally captures the workspace `analyze!` /
`synthesize!` paths of the CT and STICT into CUDA graphs, replacing their many small kernel launches
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
