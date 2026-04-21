# Julia Long-Test Workflow Skill

This skill describes how to run Contourlets.jl tests, benchmarks, and quality
checks in longer-running contexts (CI, interactive sessions, pre-commit hooks).

## Running Tests

### All tests (filtered to this package)
```bash
cd /path/to/Contourlets
julia --project=. --startup-file=no -e '
using TestItemRunner
@run_package_tests filter=ti -> contains(ti.filename, "Contourlets/test/") verbose=true
'
```

### Single test file
```bash
julia --project=. --startup-file=no -e '
using TestItemRunner
@run_package_tests filter=ti -> contains(ti.filename, "test_transforms") verbose=true
'
```

### By tag
```bash
julia --project=. --startup-file=no -e '
using TestItemRunner
@run_package_tests filter=ti -> :ct in ti.tags verbose=true
'
```

### Quality only (Aqua + JET)
```bash
julia --project=. --startup-file=no -e '
using TestItemRunner
@run_package_tests filter=ti -> :quality in ti.tags verbose=true
'
```

## Running Benchmarks

### Quick smoke-test (runs all SUITE entries once)
```bash
julia --project=benchmark --startup-file=no -e '
include("benchmark/benchmarks.jl")
using BenchmarkTools
results = run(SUITE; verbose=true, samples=3, seconds=5)
BenchmarkTools.save(".temp/benchmark_results.json", results)
'
```

### Comparison between two revisions (AirspeedVelocity)
```bash
benchpkg Contourlets \
    --rev=master,HEAD \
    --script=benchmark/benchmarks.jl \
    --output-dir=.temp/bench \
    --exeflags="--threads=2"
```

## JET Triage Steps

If `JET.test_package` reports inference failures:

1. Identify the failing function from the JET error message.
2. Run `JET.@report_opt f(args...)` interactively to see the inference chain.
3. Common fixes:
   - Add `T = float(eltype(x))` and use `T.(filter_vec)` to pin the output type.
   - Replace `Vector{Any}` with `Vector{Matrix{T}}` in struct fields.
   - Avoid `eltype(::Type{AbstractArray})` — use a concrete type parameter.
4. Re-run `JET.test_package` after each fix to confirm resolution.

## Artifact Storage

All generated outputs (coverage, benchmark JSON, profiling data) go into
`.temp/` which is git-ignored:
```
.temp/
├── benchmark_results.json
├── coverage/
└── profiles/
```
