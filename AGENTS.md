# AGENTS.md — Contourlets.jl

This file guides AI agents (and human contributors) through the repository
structure and the conventions that are *specific to this package*. General
Julia best practices live in the skills under `.agents/skills/` (see below) and
are not repeated here.

## API Usage Pattern

```julia
using Contourlets

img = randn(256, 256)
params = ContourletParams(J=4, L_array=parabolic_levels(4))

# Allocating API
coeffs = ct_forward(img, params)
img_rec = ct_inverse(coeffs, params)

# Preallocated-buffer API (for iterative algorithms)
ws = make_workspace(params, size(img))
coeffs = similar_coefficients(params, size(img))
img_out = similar(img)
for iter in 1:100
    ct_forward!(coeffs, img, params; workspace=ws)
    ct_inverse!(img_out, coeffs, params; workspace=ws)
end
```

`nsct_forward`/`nsct_inverse` (and `!` variants) follow the same pattern via
`make_nsct_workspace`. See "In-place companions" below for the naming contract.

The 3-D transform is a parallel API (`ContourletParams3`, `nsct3_forward`,
`make_nsct3_workspace`), not a dimensional generalisation — see "3-D stack"
below. It adds a **streaming** form, because the materialised coefficient set
does not fit at working sizes:

```julia
vol    = randn(Float32, 256, 256, 256)
params = ContourletParams3(J=3, L_array=[2, 2, 3])   # 48 directional subbands
ws     = make_nsct3_workspace(params, size(vol), Float32)

coarse = nsct3_forward(vol, params; workspace=ws) do j, k, band
    # `band` is valid only until this returns
end
```

## Quick Start

1. **Read this file** for repository layout and project-specific invariants.
2. **Load the matching skill** from `.agents/skills/` for the kind of work you
   are doing (performance, benchmarking, JET, packaging, docs, GPU, threading).
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
| `my-gpu-ext` | Adding/modifying GPU kernels, primitives, or CUDA graph capture in `ext/`. |
| `my-threading` | Adding threaded loops, modifying `ThreadingPolicy` dispatch, or diagnosing threading performance. |
| `my-localcoverage` | Measuring coverage percentages, generating HTML reports, enforcing a minimum coverage target. |

The first five cover generic Julia patterns (type stability, `@inbounds`, column-major
loops, `$`-interpolation in benchmarks, test-only deps in `test/Project.toml`,
the `[workspace] projects = ["test"]` layout, never editing `Manifest.toml`,
`JULIA_PKG_SERVER_REGISTRY_PREFERENCE=eager`, etc.). Don't duplicate them here.
The last three are project-specific and cover invariants that are unique to this package.

---

## Repository Layout

```
Contourlets/
├── src/                       # Package source (Julia)
│   ├── Contourlets.jl         # Main module — all exports here
│   ├── types.jl               # Core types: FilterPair, ContourletParams, …
│   ├── types3.jl              # 3-D types: ContourletParams3, NSCT3Coefficients
│   ├── workspace.jl           # ContourletWorkspace + make_workspace
│   ├── precompile.jl          # PrecompileTools workload
│   ├── filters/
│   │   ├── cdf97.jl           # CDF 9/7 LP filter constants
│   │   ├── q2345.jl           # "23-45" Phoong (pkva) ladder DFB filter constants
│   │   ├── bernstein.jl       # Bernstein mapping kernels for the 3-D hourglass FB
│   │   └── filter_utils.jl    # upsample_filter, check_pr_condition
│   ├── primitives/
│   │   ├── conv2d.jl          # 2-D separable convolution (direct + FFTW)
│   │   ├── conv3d.jl          # 3-D separable convolution (direct)
│   │   ├── conv3d_fft.jl      # 3-D periodic convolution via FFT (auto-selected)
│   │   ├── sampling.jl        # rect up/downsampling
│   │   ├── quincunx.jl        # quincunx lattice up/downsampling
│   │   └── shearing.jl        # shear! / inv_shear!
│   ├── pyramid/
│   │   ├── laplacian_pyramid.jl       # lp_decompose / lp_reconstruct
│   │   ├── nonsubsampled_pyramid.jl   # nsp_decompose / nsp_reconstruct
│   │   └── nonsubsampled_pyramid3.jl  # nsp3_decompose / nsp3_reconstruct
│   ├── directional/
│   │   ├── quincunx_fb.jl     # 2-channel quincunx filter bank
│   │   ├── dfb.jl             # L-level DFB binary tree
│   │   ├── nsdfb.jl           # Non-subsampled DFB
│   │   ├── hourglass3.jl      # 3-D nonsubsampled hourglass FB (3 subbands)
│   │   └── nsdfb3.jl          # 3-D NSDFB: hourglass + wedges (3·2^L subbands)
│   └── transforms/
│       ├── ct.jl              # ct_forward / ct_inverse / ct_forward! / ct_inverse!
│       ├── nsct.jl            # nsct_forward / nsct_inverse / ! variants
│       ├── nsct3.jl           # nsct3_forward / nsct3_inverse + NSCT3Workspace
│       └── surfacelet3.jl     # decimated 3-D transform (frequency-domain boxes)
├── test/
│   ├── runtests.jl            # TestItemRunner entry point
│   └── items/
│       ├── test_filters.jl
│       ├── test_primitives.jl
│       ├── test_pyramid.jl
│       ├── test_dfb.jl
│       ├── test_transforms.jl
│       ├── test_workspace.jl
│       ├── test_3d.jl         # conv3d + NSP3, tags :primitives / :pyramid
│       ├── test_hourglass3.jl # Bernstein kernels + hourglass FB
│       ├── test_nsdfb3.jl     # 3-D directional filter bank
│       ├── test_nsct3.jl      # 3-D NSCT, tag :nsct3
│       ├── test_surfacelet3.jl # decimated 3-D transform, tag :surfacelet3
│       ├── test_surfbox_reference.jl # golden data from SurfBox, tag :directional
│       ├── test_gpu.jl        # universal GPU tests (GPUEnv + JLArrays), tag :gpu
│       └── test_quality.jl    # Aqua checks
│   └── reference/             # SurfBox golden data + MATLAB generator scripts
├── benchmark/
│   └── benchmarks.jl          # BenchmarkTools SUITE
├── docs/
│   ├── make.jl
│   └── src/
│       ├── index.md
│       ├── theory.md
│       ├── theory3d.md        # 3-D constructions and where they leave the papers
│       ├── api.md
│       └── examples/
│           ├── ct_example.md
│           ├── nsct_example.md
│           └── nsct3_example.md
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
- **Static analysis.** Run ReLint after Runic before committing:
  ```bash
  runic src/
  JULIA_LOAD_PATH="@relint:@stdlib" julia --startup-file=no scripts/relint.jl src/
  ```
  `scripts/relint.jl` suppresses rules that are false-positives for this library
  (`@inbounds`, `return type annotation`, `in`/`tin`, `unsafe-logging`).
  See `scripts/README.md` for one-time setup instructions.

---

## Testing Conventions

- All tests are self-contained `@testitem` blocks (TestItemRunner): each block
  must import all dependencies except `Contourlets` and `Test` inside the block.
- Tag taxonomy: `:filters`, `:primitives`, `:pyramid`, `:directional`, `:ct`,
  `:nsct` (also implies `:ct`), `:nsct3` (the 3-D transform), `:surfacelet3`
  (its decimated counterpart), `:quality` (Aqua/JET), `:docs`, `:gpu`.
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
- **3-D stack: parallel types, frequency-domain directional stage.** The 2-D
  types are pinned to `AbstractMatrix` through the workspace and the GPU
  extension, and nothing in the directional stage carries over — the quincunx
  lattice, the shear index remap and the Q2345 ladder are all planar, and Lu &
  Do prove hourglass partitioning cannot be critically sampled above 2-D. So
  `ContourletParams3` / `NSCT3Coefficients` / `NSCT3Workspace` are parallel to
  their 2-D counterparts rather than generalisations. The 3-D directional bank
  is built as frequency-domain masks (exact for the periodic, nonsubsampled
  case, and it avoids the FIR truncation the papers need for streaming video);
  perfect reconstruction is *structural* — the weights are a partition of unity,
  so synthesis is either a plain sum or the same masks reapplied, depending on
  `tight` (see below). Consequence worth remembering: **PR alone proves
  nothing about wedge geometry.** A wrong directional partition still
  reconstructs perfectly, which is how a 2-D `nsdfb` regression once slipped
  through. Directional tests must assert the *index* of the subband a plane wave
  lands in.
- **3-D periodic filtering goes through the FFT by default.** Periodic separable
  filtering is circular convolution, so an FFT evaluation is exact, not an
  approximation, and its cost is independent of tap count — which matters
  because à trous filters grow as `2^(j-1)` per axis. `conv3d_sep!`
  auto-selects it (`_conv3d_use_fft`); complex volumes always take it, since the
  direct kernel has no interior fast path for them. At 64³ Float32 with 33 taps:
  182 ms direct vs 1.8 ms via FFT.
- **Complex is two real passes, everywhere in 3-D.** `plan_rfft` is real-only,
  so a complex volume is filtered as its real part then its imaginary part
  through the identical kernel. Two real transforms cost about one complex
  transform, and it keeps `T(x + i·y) == T(x) + i·T(y)` true *bit-for-bit* —
  which the test suite asserts with `== 0.0`, and which is a free correctness
  check for code built on top.
- **GPU = whole transform on the device** — the `ContourletsGPUExt` extension
  runs every stage of CT/NSCT on the device. Pyramid functions are reused
  unchanged (broadcast-based dispatch picks up GPU primitives automatically); both
  directional banks are kernelised (`dfb_gpu.jl`, `nsdfb_gpu.jl`). Coefficients
  stay device-resident (`ContourletCoefficients{Td,A}`, `A<:AbstractMatrix`); use
  `Array(·)` / `Adapt.adapt(Array, coeffs)` to bring them back. `ContourletsCUDAExt`
  adds CUDA graph capture for the workspace paths. Load the `julia-gpu-ext` skill
  before touching any file under `ext/`.
- **3-D on GPU needs no kernels.** The 3-D transforms are FFT plus a pointwise
  mask, and every mask is stored as *broadcastable factors* — the mapping
  responses and wedge planes are 3-D arrays with singleton axes where they do
  not depend on a frequency — so mask application, the complex real/imaginary
  split, and the pyramid's frequency multiply are all plain broadcasts. Keep it
  that way: introducing scalar indexing into `_fill_mask!`,
  `_apply_hourglass_mask!` or `_apply_freq_filter!` silently breaks the device
  path. `ext/ContourletsGPUExt/nsct3_gpu.jl` therefore contains only
  `Adapt.adapt_structure` methods, which move the data and replan the FFTs
  (plans cannot be adapted). A workspace adapt moves the shared hourglass
  response **once**, not once per scale. JLArrays is excluded from the 3-D GPU
  tests: it has no FFT implementation, so `plan_rfft` falls back to a
  scalar-indexing generic. CUDA and OpenCL are verified.
- **The 3-D directional stages are validated against SurfBox.** SurfBox (Lu and
  Do's reference implementation, MATLAB File Exchange 14485) is available and
  runs under `module load matlab`. `test/reference/` holds golden data extracted
  from it plus the MATLAB scripts that regenerate it, and
  `test/items/test_surfbox_reference.jl` checks this package against it. The
  agreement is on *geometry*, not coefficients: 94.6% of frequencies pick the
  same dominant hourglass (every disagreement a near-tie), and across 80 plane
  waves the wedge labellings are related by a bijection with the reference's
  choice always this package's top-2. One known deviation is recorded there and
  in `docs/src/theory3d.md`: `surfacelet3`'s redundancy grows with the direction
  count (15.4 at 24 subbands, 46.4 at 96) where SurfBox stays flat at 3.93,
  because axis-aligned support boxes recover the radial but not the directional
  decimation.
- **The 3-D directional bank is a Parseval frame by default.** The three
  hourglass weights are a partition of unity for any `order`/`poly`, and so are
  the wedge weights, so the products are too. `tight = true` (the default on
  `make_hourglass3` / `make_nsdfb3` / `ContourletParams3`) takes the masks to be
  `√W`, which makes `Σ V² = 1` identically — the pointwise normalisation SurfBox
  uses, scaled to bound 1 instead of `N`. Analysis then preserves energy, which
  is what thresholding and iterative reconstruction want. The price is
  synthesis: the masks have to be reapplied, so the directional inverse costs a
  forward transform per subband plus one inverse instead of a bare sum. Pass
  `tight = false` to get the old partition-of-unity bank back, with a free
  inverse and a frame-bound ratio of 3.
  **The normalisation makes mask evenness load-bearing.** A plain sum hides an
  asymmetric mask — `Σ W_k X = X` regardless — but reapplying `√W` requires each
  subband to survive an `irfft`/`rfft` round trip, which requires the mask to be
  even on the self-conjugate planes of the rfft grid. It was not, and PR failed
  by 0.27 with no error. `_wedge_ratio` now fixes the sign from the first
  coordinate that can carry one; `nsdfb3 — masks are even in frequency` pins it.
- **Hybrid Threading Architecture** — real data (`Float32`/`Float64`) uses
  `LoopVectorization.@turbo` (SIMD, single thread); complex data uses
  `Polyester.@batch` (multi-core, near-zero spawn overhead). Public API exposes
  `threading::ThreadingPolicy` kwargs: `Auto` (default), `Enabled`, `Disabled`.
  `Auto` enables threading for complex types only. Load the `julia-threading` skill
  before adding or modifying threaded loops.
