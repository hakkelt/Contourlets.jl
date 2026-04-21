# AGENTS.md — Contourlets.jl

This file guides AI agents (and human contributors) through the repository
structure and working conventions.

## Quick Start for Agents

1. **Read this file** to understand the repository layout.
2. **Match the edited path** to the relevant instruction file below.
3. **Use the skill file** for long-running commands (tests, benchmarks, JET).

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
│   │   ├── q2345.jl           # Haar quincunx DFB filter constants
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
└── .github/
    ├── workflows/             # CI.yml, documentation.yml, CompatHelper.yml, TagBot.yml
    ├── instructions/          # Per-path coding conventions
    └── skills/
        └── julia-long-test-workflow/SKILL.md
```

---

## Instruction Files by Path

| Editing path | Read this instruction file |
|---|---|
| `src/**/*.jl` | `.github/instructions/julia-contourlet-impl.instructions.md` |
| `test/**/*.jl` | `.github/instructions/julia-testing-and-quality.instructions.md` |
| `src/**/*.jl`, `benchmark/**/*.jl` | `.github/instructions/julia-performance.instructions.md` |

---

## Key Invariants

These must remain true after any change:

| Invariant | Verification command |
|---|---|
| CT perfect reconstruction `< 1e-12` | `test_transforms.jl` — "CT PR" test items |
| NSCT perfect reconstruction `< 2e-15` | `test_transforms.jl` — "NSCT PR" test items |
| Workspace path gives same result as allocating | `test_workspace.jl` |
| No Aqua regressions | `test_quality.jl` |
| All 119 tests pass | Run all test items |

---

## Long-Running Commands

See `.github/skills/julia-long-test-workflow/SKILL.md` for:
- Running specific test subsets by tag or filename
- Benchmark comparison with AirspeedVelocity
- JET inference-failure triage steps
- Artifact storage conventions (`.temp/`)

---

## Design Decisions

- **No downsampling in NSCT** — all subbands have the same spatial size as the input.
- **Duck-typed public API** — functions accept `AbstractMatrix` without `where T`
  constraints that cause dispatch issues.  Float type is pinned via
  `T_out = float(eltype(image))` inside the function body.
- **Workspace buffers use views** — `@view ws.tmp_buf[1:n1, 1:n2]` passes a
  correctly-sized scratch buffer at each LP level without re-allocation.
- **FFTW.MEASURE plans** — cached in `_PLAN_CACHE` keyed on `(src_size, kernel_size)`.
- **Test-only deps in main Project.toml** — Aqua and TestItemRunner are listed
  in `[deps]` (not `[extras]`) to allow running tests from the root environment.
  This requires `stale_deps=false` in the Aqua test call.
