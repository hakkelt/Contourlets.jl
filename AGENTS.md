# AGENTS.md — Contourlets.jl

This file guides AI agents (and human contributors) through the repository
structure and the conventions that are *specific to this package*. General
Julia practice is not repeated here. The package-specific skills, under
`.agents/skills/`, are `my-gpu-ext` and `my-threading`.

## Package family

```
MultiscaleFilterBanks.jl   (shared core: FilterPair, CDF97, à trous filters, threading, mask-bank engine)
        ├── Contourlets.jl   (this package — 2-D CT / NSCT)
        ├── Surfacelets.jl   (the surfacelet transform, for volumes)
        └── Shearlets.jl     (2-D and 3-D shearlets)
```

This package is planar throughout. Volumes live in
[Surfacelets.jl](https://github.com/hakkelt/Surfacelets.jl), which is **not**
this transform with an extra axis: the quincunx lattice, the shear index remap
and the Q2345 ladder are all planar, and Lu and Do prove that the hourglass
frequency partition a volume needs cannot be critically sampled at all. The two
packages share only the pyramid filter machinery.

The shared surface with the core is small and explicit: `FilterPair`, `CDF97`,
`upsample_filter`, the threading policy types, and the internals `_data_eltype`,
`_filter_eltype`, `_use_threading`, the scoped FFTW thread count
(`_with_fftw_threads`), `_clamp_idx`, `_nsp_atrous_filters`, and the
pyramid's frequency responses (`_nsp_responses`, `_centred_dtft`), and the
mask-bank engine that runs the NSCT and TICT workspaces (`_MaskBankEngine`,
`_MaskItem`, `_mask_bank_forward!`, `_mask_bank_inverse!`; this package adds
`_subband` methods for its coefficient sets in `src/workspace/mask_bank.jl`).
All are re-exported (public names) or imported (internals) in
`src/Contourlets.jl`.

**First checkout**: CI resolves `MultiscaleFilterBanks.jl` from the registry. To
develop against a sibling checkout, run `julia --project=. dev/bootstrap.jl`
once; it `Pkg.develop`s the checkout into every environment of the workspace.

## Quick Start

1. **Read this file** for project-specific invariants.
2. **Load the matching skill** for GPU or threading work (`my-gpu-ext`,
   `my-threading`).
3. **Honour the invariants** below — every change must keep them.

## Commit Rules

- **Commit Message Formatting**: Follow the "Conventional Commits" format with the structure `type(scope): subject` and a detailed description (including performance numbers or context) separated by a blank line. For simple commits affecting only a couple lines, a detailed description is optional. Always append co-author line referencing yourself.
- **Committing Changes**: ONLY commit code upon explicit instruction from the user.

---


## Implementation Conventions

Conventions that are non-obvious or specific to this package's API contract:

- **Filter storage.** `FilterPair{T}` stores `h` (analysis) and `g` (synthesis)
  as `Vector{T}`. `QuincunxFilterPair{T}` stores only the 1-D ladder filter
  `f_ladder::Vector{T}` (Phoong's β, unmodulated); every directional filter is
  derived from it. Filter constants are defined once in `src/filters/` and
  referenced by value — never rebuild filter arrays inside hot loops.
- **Element types: real filters, real-or-complex data.** `FilterPair` and the
  `_data_eltype`/`_filter_eltype` helpers come from MultiscaleFilterBanks.
  Filters are always real
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
  `f!(dst, args..., params; workspace=nothing)` writes into `dst`, and
  `f(args..., params; workspace=nothing)` allocates and returns the result — the
  **only** difference is whether the caller supplies the output container.  The
  transforms that thread (NSCT, TICT) also take `threading=Auto()` on both
  variants; the CT and the STICT have nothing to thread and take no `threading`
  keyword.  Both
  variants accept the same optional `workspace` kwarg for reusing scratch buffers; pass a `CTWorkspace` from
  `make_workspace` (or an `NSCTWorkspace` from `make_nsct_workspace`) to eliminate per-call allocation in the
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
  (`@inbounds`, `return type annotation`, `in`/`tin`, `haskey`/`thaskey`, `unsafe-logging`).
  See `scripts/README.md` for one-time setup instructions.

---

## Testing Conventions

- All tests are self-contained `@testitem` blocks (TestItemRunner): each block
  must import all dependencies except `Contourlets` and `Test` inside the block.
- Tag taxonomy: `:filters`, `:primitives`, `:pyramid`, `:directional`, `:ct`,
  `:nsct` (also implies `:ct`), `:tict` (TICT and STICT), `:quality` (Aqua/JET),
  `:jet`, `:docs`, `:gpu`.
- Seed every random input (`Random.seed!(42)`); never rely on global RNG state.
- NSCT shift-invariance is a required property — compare `nsct_forward(x)` with
  `nsct_forward(circshift(x, s))` against `circshift(., s)` (tol `1e-10`).

---

## Long-Running Commands

Run from the package root; the commands below are the project entry points.

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
  NSCT is exactly invariant under circular shifts of the input.  Periodic
  filtering is a product on the DFT grid, so an NSCT workspace holds the whole
  pyramid as masks (`_nsp_masks`): `nsct_forward!` is one FFT plus one inverse
  FFT per output, no convolution.  The allocating path and the public
  `nsp_decompose` still convolve (skipping the à trous zeros); both agree to
  rounding.
- **Perfect reconstruction proves nothing about directional geometry.** The
  directional stage reconstructs exactly by construction — Phoong's quincunx pair
  satisfies the PR conditions for any ladder filter — so a wrong directional
  partition still round-trips perfectly. That is how an `nsdfb` regression once
  slipped through (sibling wedges merged from `l = 4`). Directional tests must
  assert the *index* of the subband a plane wave lands in, not just PR.
- **The DFB is the thesis' multichannel bank, computed on the DFT grid.** Each
  channel is `yₖ[n] = (hₖ ⊛ x)[Sₖn]` with the closed-form equivalent filter `Hₖ`
  (Do's thesis eqs. 3.17–3.23) and a diagonal `Sₖ`; the NSDFB is the same `Hₖ`
  without decimation. It equals the MATLAB toolbox at `l ≤ 2` and deliberately
  not from `l = 3` (the toolbox is not equivariant on its lattices there). Keep
  the bank paths broadcast + FFT only; no scalar indexing, no allocation inside
  `_bank_analyze!` / `_bank_synthesize!` (CUDA graph capture relies on it).
- **NSDFB coarse-scale upsampling is measured, not guessed.** da Cunha et al.
  (§II-A.3) filter pyramid level `j` with `Uₖ(z^(2^m I))` but leave `m` open;
  `m = max(0, j − 2)` minimises the energy each NSP level leaks outside the
  ideal wedge for every depth 1–5. The TILP is different (it keeps the
  decimated LP's aliasing), and there `m = 0` leaks least, so the TICT does not
  upsample. Re-measure before changing either rule.
- **TICT/STICT are Eslami & Radha's, on the DFT grid.** The TILP's four highpass
  channels per level are closed-form masks (`tilp.jl`) — the decimated LP's
  polyphase channels for every lattice shift, which a test checks against an
  independent pyramid. Channels 2 and 3 use half trees; the fan each splits is
  fixed by the channel's energy (`_TILP_HALVES`), not by a label in the paper.
- **Hybrid Threading Architecture** — real data (`Float32`/`Float64`) uses
  `LoopVectorization.@turbo` (SIMD, single thread); complex data uses
  `Polyester.@batch` (multi-core, near-zero spawn overhead). The NSCT and
  TICT transforms (and `conv2d`) expose `threading::ThreadingPolicy` kwargs:
  `Auto` (default), `Enabled`, `Disabled`; the CT and STICT do not.
  `Auto` enables threading for complex types only. Load the `my-threading` skill
  before adding or modifying threaded loops.  The FFT-bound workspace paths
  (NSCT, TICT) thread differently: one `@batch` task per thread
  (MultiscaleFilterBanks' `_foreach_chunk`/`_chunked_sum!`) runs a
  contiguous chunk of the independent subband FFTs with its own buffer (and,
  for the inverse, its own partial sum).  FFTW's own threads split one FFT and
  gain ~nothing at 256²; this gains ~3.5× on 4 threads.  Plans stay
  single-threaded, and the choice is per call.  Never call
  `FFTW.set_num_threads` directly: plan inside `_with_fftw_threads`, which
  restores the count and yields to a caller's NestedThreading budget.
- **Same-size products use FastBroadcast's `@..`.** Base's broadcast checks every
  axis for a singleton at run time, which keeps the mask products of the
  workspace paths from vectorising (~2× slower).  `@..` fails on singleton axes
  and falls back to Base's broadcast for device arrays.
