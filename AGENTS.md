# AGENTS.md — Contourlets.jl

This file guides AI agents (and human contributors) through the repository
structure and the conventions that are *specific to this package*. General
Julia best practices live in the skills under `.agents/skills/` (see below) and
are not repeated here.

## Quick Start

1. **Read this file** for repository layout and project-specific invariants.
2. **Load the matching skill** from `.agents/skills/` for the kind of work you
   are doing (performance, benchmarking, JET, packaging, docs).
3. **Honour the invariants** in the table below — every change must keep them.

## Commit Rules

- **Commit Message Formatting**: Follow the "Conventional Commits" format with the structure `type(scope): subject` and a detailed description (including performance numbers or context) separated by a blank line. For simple commits affecting only a couple lines, a detailed description is optional. Always append co-author line referencing yourself. Do **not** add a `Claude-Session:` trailer line.
- **Committing Changes**: ONLY commit code upon explicit instruction from the user.

---

## Skills (`.agents/skills/`)

Vendor-neutral skills. Load the one that matches the task and open only the
reference file it points you to.

| Skill | Use when |
|---|---|
| `julia-perf` | Diagnosing slow code, reducing allocations, fixing type instabilities. |
| `julia-bench` | Writing/running benchmarks, comparing revisions, benchmark CI. |
| `julia-jet` | Running JET inference/type analysis on the package. |
| `julia-package-dev` | Environments, dependencies, extensions, multi-package workspace. |
| `julia-docs` | Documenter.jl site, docstrings, doctests, citations. |

These cover the generic patterns (type stability, `@inbounds`, column-major
loops, `$`-interpolation in benchmarks, test-only deps in `test/Project.toml`,
the `[workspace] projects = ["test"]` layout, never editing `Manifest.toml`,
`JULIA_PKG_SERVER_REGISTRY_PREFERENCE=eager`, etc.). Don't duplicate them here.

---

## Repository Layout

```
Contourlets/
├── src/                       # Package source (Julia)
│   ├── Contourlets.jl         # Main module — all exports here
│   ├── types.jl               # Core types: FilterPair, ContourletParams, …
│   ├── workspace.jl           # ContourletWorkspace + make_workspace
│   ├── precompile.jl          # PrecompileTools workload
│   ├── filters/
│   │   ├── cdf97.jl           # CDF 9/7 LP filter constants
│   │   ├── q2345.jl           # "23-45" Phoong (pkva) ladder DFB filter constants
│   │   └── filter_utils.jl    # upsample_filter, check_pr_condition
│   ├── primitives/
│   │   ├── conv2d.jl          # 2-D separable convolution (direct + FFTW)
│   │   ├── sampling.jl        # rect up/downsampling
│   │   ├── quincunx.jl        # quincunx lattice up/downsampling
│   │   └── shearing.jl        # shear! / inv_shear!
│   ├── pyramid/
│   │   ├── laplacian_pyramid.jl      # lp_decompose / lp_reconstruct
│   │   └── nonsubsampled_pyramid.jl  # nsp_decompose / nsp_reconstruct
│   ├── directional/
│   │   ├── quincunx_fb.jl     # 2-channel quincunx filter bank
│   │   ├── dfb.jl             # L-level DFB binary tree
│   │   └── nsdfb.jl           # Non-subsampled DFB
│   └── transforms/
│       ├── ct.jl              # ct_forward / ct_inverse / ct_forward! / ct_inverse!
│       └── nsct.jl            # nsct_forward / nsct_inverse / ! variants
├── test/
│   ├── runtests.jl            # TestItemRunner entry point
│   └── items/
│       ├── test_filters.jl
│       ├── test_primitives.jl
│       ├── test_pyramid.jl
│       ├── test_dfb.jl
│       ├── test_transforms.jl
│       ├── test_workspace.jl
│       ├── test_gpu.jl        # universal GPU tests (GPUEnv + JLArrays), tag :gpu
│       └── test_quality.jl    # Aqua checks
├── benchmark/
│   └── benchmarks.jl          # BenchmarkTools SUITE
├── docs/
│   ├── make.jl
│   └── src/
│       ├── index.md
│       ├── theory.md
│       ├── api.md
│       └── examples/
│           ├── ct_example.md
│           └── nsct_example.md
├── .agents/skills/            # Vendor-neutral Julia skills (see table above)
└── .github/workflows/         # CI.yml, documentation.yml, CompatHelper.yml, TagBot.yml
```

---

## Implementation Conventions

Conventions that are non-obvious or specific to this package's API contract:

- **Filter storage.** `FilterPair{T}` stores `h` (analysis) and `g` (synthesis)
  as `Vector{T}`. `QuincunxFilterPair{T}` stores `h_q`, `g_q` as `Matrix{T}`
  (1×N by convention). Filter constants are defined once in `src/filters/` and
  referenced by value — never rebuild filter arrays inside hot loops.
- **Element types: real filters, real-or-complex data.** Filters are always real
  (`FilterPair`/`QuincunxFilterPair`/`ContourletParams` are `Tf <: AbstractFloat`);
  image data may be complex (`Td <: Number`). The two flow independently —
  `Td = _data_eltype(image)`, `Tf = _filter_eltype(Td) = real(float(Td))` — so a
  complex image is filtered as complex·real (never complex·complex). Multiply
  kernels (`conv2d_sep!`, `conv2d` direct, `_sefilter2`, `_nsqfb_*`, `qfb`) take
  `src::{Td}` / filter `::{Tf}` separately and accumulate in `Td`; coefficient and
  workspace containers are 2-param `{Td, Tf}` (dispatch annotations `{T}` bind the
  first param = data type). The transform is real-linear, so `T(x + i·y)` equals
  `T(x) + i·T(y)` bit-for-bit — a useful correctness invariant. The FFTW conv
  backend is real-only; complex data routes through the direct backend.
- **In-place companions.** The top-level transforms follow the convention that
  `f!(dst, args..., params; workspace=nothing, threading=Auto())` writes into
  `dst`, and `f(args..., params; workspace=nothing, threading=Auto())` allocates
  and returns the result — the **only** difference is whether the caller supplies
  the output container.  Both variants accept the same optional `workspace` kwarg
  for reusing scratch buffers; pass a `ContourletWorkspace` from
  `make_workspace`/`make_nsct_workspace` to eliminate per-call allocation in the
  pyramid stage.  Do not allocate inside the `_ct_forward_ws!` /
  `_ct_inverse_ws!` / `_nsct_forward_ws!` / `_nsct_inverse_ws!` workspace paths.
- **Duck-typed public API.** Functions accept `AbstractMatrix` with no
  `where T` constraints (those cause dispatch issues here). Pin the float type
  inside the body via `T_out = float(eltype(image))` and convert filters with
  `T_out.(fp.h)`, not `convert.(T_out, fp.h)`.
- **Boundary modes.** Default `:symmetric` (mirror); `:periodic` also
  supported. Pass as `Val(:symmetric)` / `Val(:periodic)` for compile-time
  dispatch rather than a runtime branch.
- **Shearing is a pure index remap** (no arithmetic on values):
  horizontal `(i, j) → (i, mod1(j + i, n2))`,
  vertical `(i, j) → (mod1(i + j, n1), j)`; the inverse flips the sign in the
  modular step.
- **FFTW plans: no Julia-side plan cache.** FFTW already caches `MEASURE`
  wisdom (the expensive timing step) globally in the C library, so re-planning a
  previously-seen size is sub-millisecond. A `Dict` of plan objects only adds a
  lock and a type-unstable cache. Use `FFTW.MEASURE` for repeated transforms,
  `FFTW.ESTIMATE` when a plan is used once and first-call latency matters.
- **Formatting.** Runic is mandatory — run `runic src/` before committing.

---

## Testing Conventions

- All tests are self-contained `@testitem` blocks (TestItemRunner): each block
  must import all dependencies except `Contourlets` and `Test` inside the block.
- Tag taxonomy: `:filters`, `:primitives`, `:pyramid`, `:directional`, `:ct`,
  `:nsct` (also implies `:ct`), `:quality` (Aqua/JET), `:gpu`.
- Seed every random input (`Random.seed!(42)`); never rely on global RNG state.
- NSCT shift-invariance is a required property — compare `nsct_forward(x)` with
  `nsct_forward(circshift(x, s))` against `circshift(., s)` (tol `1e-10`).

---

## Long-Running Commands

Run from the package root. For general benchmarking/JET workflow see the
`julia-bench` / `julia-jet` skills; the commands below are the project entry
points.

**Tests**:
Run tests through `test/runtests.jl`. It supports filtering by name and tag using a comma-separated argument.

```bash
# Run all tests
julia --project -e 'using Pkg; Pkg.test()'
# OR
julia --project=test test/runtests.jl

# Run focused tests by passing tags (starting with :) or names
julia --project=test test/runtests.jl :ct,:gpu
julia --project=test test/runtests.jl "make_nsct_workspace (type-first positional API)"
```

**Benchmarks** — smoke test, then revision comparison:
```bash
julia --project=benchmark --startup-file=no -e '
include("benchmark/benchmarks.jl"); using BenchmarkTools
BenchmarkTools.save(".temp/benchmark_results.json", run(SUITE; verbose=true, samples=3, seconds=5))
'
benchpkg Contourlets --path . --rev=master,HEAD --script=benchmark/benchmarks.jl \
    --output-dir=.temp/bench --exeflags="--threads=2"
benchpkgtable Contourlets --path . --rev=master,HEAD --input-dir=.temp/bench --ratio
```

**Artifacts** — all generated output (test run outputs, coverage, benchmark JSON, profiles) goes
under `.temp/`, which is git-ignored to avoid re-runing failing long-running commands for full
output.

---

## Design Decisions

- **No downsampling in NSCT** — all subbands have the same spatial size as the
  input. All NSP/NSDFB filtering uses periodic (circular) convolution, so the
  NSCT is exactly invariant under circular shifts of the input.
- **GPU = whole transform on the device** — the `ContourletsGPUExt` extension
  runs every stage of CT/NSCT on the device. The LP/NSP pyramid functions are
  *reused unchanged* from the main package: their allocating methods are
  broadcast-based and dispatch to the GPU primitives (separable convolution,
  sampling, shearing) when given device arrays, so there are no GPU-specific
  pyramid overloads. Both directional banks are kernelised too: the decimated
  DFB via GPU `_resamp`/`_sefilter2` (`dfb_gpu.jl`) plus the type-generic
  polyphase tree, and the NSDFB via per-pixel `_nsqfb_*` kernels (`nsdfb_gpu.jl`).
  Each device kernel reproduces the CPU reduction order, so GPU
  `ct_forward`/`nsct_forward` match the CPU path (to Float32 precision). Results
  stay on the device: the coefficient containers carry a storage-type parameter
  (`ContourletCoefficients{Td,A}`, `A<:AbstractMatrix`), so a GPU forward
  returns device-resident coeffs and `ct_inverse(coeffs, params)`
  reconstructs on the device. Use `Array(·)` / `Adapt.adapt` to move coeffs across. Complex device
  arrays work too: the GPU kernels keep filters real (`real(T)`) and accumulate
  in the data type, mirroring the CPU split. Shared src must stay
  device-portable — allocate scratch with `_zeros_like`/`similar` (not `zeros`),
  avoid index-vector gathers (use `circshift`), and size work vectors from the
  actual array type, not `Matrix{T}`. Tests use GPUEnv.jl on JLArrays (CI) and
  any real backend present (`:gpu` tag).
- **Hybrid Threading Architecture** — CPU performance utilizes a dual-path 
  architecture to achieve maximum performance on both `Real` and `Complex` 
  data. For real data (`Float32`/`Float64`), we rely strictly on 
  `LoopVectorization.@turbo` for SIMD acceleration on a single thread (avoiding 
  costly task spawns). Because `LoopVectorization` does not support complex 
  numbers, complex data falls back to an un-vectorized inner loop which is
  parallelized across CPU cores using `Polyester.@batch`. `Polyester`'s 
  near-zero overhead persistent thread pool avoids the task-spawning latency 
  that destroys recursive performance, allowing `ComplexF64` transforms to execute 
  at virtually the same wall-clock speed as SIMD-accelerated `Float64` transforms. 
  Public API `threading::ThreadingPolicy` kwargs (`Auto`, `Enabled`, `Disabled`)
  allow users to explicitly control this behavior, with `Auto` enabling multithreading
  for complex data and disabling it for real data by default.
