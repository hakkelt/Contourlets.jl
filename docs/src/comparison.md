# [Comparison with the MATLAB toolboxes](@id comparison)

This page compares Contourlets.jl against the two reference implementations:

- **MATLAB Contourlet Toolbox** (Do & Vetterli, 2005) — [FileExchange 8837](https://www.mathworks.com/matlabcentral/fileexchange/8837-contourlet-toolbox), BSD license
- **MATLAB Nonsubsampled Contourlet Toolbox** (da Cunha et al., 2006) — [FileExchange 10049](https://www.mathworks.com/matlabcentral/fileexchange/10049-nonsubsampled-contourlet-toolbox), BSD license

These toolboxes are BSD-licensed and are therefore **not** vendored into this
MIT-licensed repository.  Clone them into `matlab/` as described in
[`matlab/README.md`](https://github.com/hakkelt/Contourlets.jl/blob/master/matlab/README.md)
to reproduce the validation and benchmarks below.

---

## Feature comparison

The 2-D transforms, against the two 2-D toolboxes; the 3-D stack is compared
against SurfBox in the [next section](@ref surfbox).

| Feature | Contourlets.jl | MATLAB CT Toolbox | MATLAB NSCT Toolbox |
|---------|---------------|------------------|-------------------|
| Language | Julia 1.10+ | MATLAB R2006+ | MATLAB R2007+ |
| Licence | MIT | BSD | BSD |
| Discrete CT (`ct_forward/inverse`) | ✅ | ✅ (`pdfbdec/pdfbrec`) | ✅ |
| Nonsubsampled CT (`nsct_forward/inverse`) | ✅ | ❌ | ✅ (`nsctdec/nsctrec`) |
| Laplacian Pyramid (standalone) | ✅ | ✅ (`lpdec/lprec`) | ✅ |
| Directional Filter Bank (standalone) | ✅ | ✅ (`fbdec/fbrec`) | ✅ |
| Preallocated-buffer workspace API | ✅ | ❌ | ❌ |
| CDF 9/7 LP filters | ✅ | ✅ | ✅ |
| "23-45" (pkva) ladder DFB filters | ✅ | ✅ | ✅ |
| Parabolic-scaling helper | ✅ | ❌ (manual) | ❌ (manual) |
| GPU support | ✅ | ❌ | ❌ |
| Float32 support | ✅ | partial (double only) | partial |
| Package manager integration | Julia Pkg | File download | File download |
| Unit tests with PR guarantees | ✅ | ❌ | ❌ |
| Type-stable, JET-verified | ✅ | N/A | N/A |
| Hybrid Multithreading (Real/Complex) | ✅ | ❌ | ❌ |

---

## [3-D: SurfBox](@id surfbox)

The 3-D reference implementation is **SurfBox** (Lu & Do,
[FileExchange 14485](https://www.mathworks.com/matlabcentral/fileexchange/14485-surfbox)),
which implements both the nonsubsampled hourglass/NDFB bank and the decimated
surfacelet transform. `test/items/test_surfbox_reference.jl` regression-tests
this package against golden data extracted from it.

| Feature | Contourlets.jl | SurfBox |
|---------|---------------|---------|
| 3-D NSCT (`nsct3d_forward/inverse`) | ✅ | ✅ (`NsdfbDec` + pyramid) |
| Surfacelet transform (`surfacelet3d_forward/inverse`) | ✅ | ✅ (`Surfdec/Surfrec`) |
| Directional stage | frequency-domain masks, exact | FIR mapping kernels (37³, 61³), truncated |
| Hourglass design | Bernstein mapping polynomials | Kaiser-windowed diamond kernel, raised to a power |
| Wedge partition | boundaries placed directly in frequency | resampled-checkerboard chain |
| Parseval normalisation | ``\sum_k V_k^2 = 1`` | ``\sum_k V_k^2 = N`` (same, scaled) |
| Surfacelet redundancy | 15.4 at 24 subbands, 46.4 at 96 | 3.93, flat in the direction count |
| Multiscale pyramid | dyadic cube shells | resamples by 3/2 per axis |
| Streaming (one subband at a time) | ✅ | ❌ |
| GPU support | ✅ (both, CUDA / OpenCL) | ❌ |

The two agree on *geometry*, not coefficients: over a 32³ grid both pick the
same dominant hourglass at 94.6% of frequencies (every disagreement a near-tie),
and across 80 plane waves the wedge labellings are related by a bijection with
the reference's choice always in this package's top two. The redundancy row is
the one real deviation, and it is structural — see
[Theory — 3-D](@ref theory3d).

---

## Algorithm correspondence

| Contourlets.jl | MATLAB CT Toolbox | MATLAB NSCT Toolbox |
|---------------|------------------|-------------------|
| `ct_forward(img, params)` | `pdfbdec(img, '9-7', 'pkva', [L1,L2,…])` | — |
| `ct_inverse(coeffs)` | `pdfbrec(coeffs, '9-7', 'pkva')` | — |
| `nsct_forward(img, params)` | — | `nsctdec(img, [L1,L2,…])` |
| `nsct_inverse(coeffs)` | — | `nsctrec(coeffs)` |
| `lp_decompose(img, CDF97)` | `lpdec(img, '9-7')` | — |
| `lp_reconstruct(c, bp, CDF97)` | `lprec(c, bp, '9-7')` | — |
| `dfb_decompose(bp, L, Q2345)` | `dfbdec(bp, 'pkva', L)` | — |
| `dfb_reconstruct(sbs, Q2345)` | `dfbrec(sbs, 'pkva')` | — |
| `nsdfb_decompose(bp, L, Q2345, tl)` | — | `nsdfbdec(bp, …)` |
| `nsdfb_reconstruct(sbs, Q2345, tl)` | — | `nsdfbrec(sbs, …)` |
| `parabolic_levels(J)` | manual `[1 2 3 4]` | manual |

---

## Numerical accuracy

Both the MATLAB toolboxes and Contourlets.jl reach perfect reconstruction (PR) in
exact arithmetic; in `Float64` the residuals sit at the round-off floor. The
values below are from the cross-validation run described under
[Benchmarks](@ref comparison-benchmarks), on random `n × n` inputs with `J = 2`,
`L_array = [2, 3]`:

| Check | Contourlets.jl | MATLAB reference |
|---|---|---|
| NSDFB, `L = 3` | 1.2e-14 … 1.4e-14 | 8.9e-15 … 1.1e-14 |
| CT, `J = 2` | 5.6e-15 … 6.0e-15 | 6.8e-14 … 8.4e-14 |
| NSCT, `J = 2` | 8.9e-15 … 1.1e-14 | 5.8e-15 … 7.5e-15 |

The NSDFB is the one stage where the two implementations use the same filters,
so its subbands can be compared directly rather than only through PR: they agree
to 7.5e-15. The CT and NSCT use different pyramid filters by design — this
package uses the JPEG 2000-normalised separable CDF 9/7 à trous pyramid, the
toolboxes a differently scaled `'9-7'` and a non-separable diamond pyramid — so
their coefficients differ and only PR is comparable.

Perfect reconstruction in the directional stage is *structural*: the Phoong et
al. (1995) two-step ladder network inverts step by step for any lifting filter,
independent of boundary handling. That is why the residuals are at the
floating-point floor rather than merely small.

---

## [Benchmarks](@id comparison-benchmarks)

All timings below were measured in a single run of `matlab/verify_vs_julia.m`,
which drives both implementations on identical inputs. Medians of 7 runs,
warm-up excluded on both sides.

!!! note "Measurement environment"
    | | |
    |---|---|
    | CPU | 2 × AMD EPYC 7352 (24 cores each, 96 logical threads) |
    | OS | Linux 4.18, x86_64 |
    | Julia | 1.13.0, 1 thread, `Float64` |
    | MATLAB | R2022a, with the toolboxes' MEX kernels compiled |
    | Host | shared HPC login node — other users' load adds noise |

    These are absolute wall-clock numbers from one machine. Re-run the script
    on your own hardware before quoting them; the ratios travel better than the
    milliseconds.

### CT — `ct_forward` / `ct_inverse` against `pdfbdec` / `pdfbrec`

| Size | Contourlets.jl fwd | MATLAB fwd | Contourlets.jl inv | MATLAB inv |
|---|---:|---:|---:|---:|
| 64 × 64 | 1.3 ms | 11.2 ms | 1.0 ms | 21.0 ms |
| 128 × 128 | 3.4 ms | 6.9 ms | 3.2 ms | 6.7 ms |
| 256 × 256 | 13.8 ms | 85.1 ms | 13.8 ms | 13.3 ms |

### NSCT — `nsct_forward` / `nsct_inverse` against `nsctdec` / `nsctrec`

The workspace column is `nsct_forward!` with a prewarmed
[`make_nsct_workspace`](@ref), which evaluates the directional stage as an FFT
and pointwise multiplies instead of spatial convolutions. It is the form to use
whenever the same size is transformed more than once; the MATLAB toolbox has no
equivalent.

| Size | Contourlets.jl fwd | + workspace | MATLAB fwd | Contourlets.jl inv | MATLAB inv |
|---|---:|---:|---:|---:|---:|
| 64 × 64 | 118 ms | 0.6 ms | 194 ms | 84 ms | 192 ms |
| 128 × 128 | 257 ms | 1.7 ms | 763 ms | 232 ms | 765 ms |
| 256 × 256 | 960 ms | 7.2 ms | 3762 ms | 839 ms | 3092 ms |

At 256² the workspace path is roughly 500× faster than the MATLAB reference and
130× faster than this package's own allocating path — the gap the
[Performance](@ref performance) page goes into.

### NSDFB — `nsdfb_decompose` / `nsdfb_reconstruct` against `nsdfbdec` / `nsdfbrec`

`L = 3` (8 directional subbands), the one stage where both implementations use
the same filters:

| Size | Contourlets.jl dec | MATLAB dec | Contourlets.jl rec | MATLAB rec |
|---|---:|---:|---:|---:|
| 64 × 64 | 43 ms | 256 ms | 43 ms | 251 ms |
| 128 × 128 | 195 ms | 660 ms | 153 ms | 764 ms |
| 256 × 256 | 491 ms | 2260 ms | 630 ms | 2274 ms |

Where this package is faster, it is mostly for structural reasons: type-stable
`Val{B}`-dispatched boundary handling with no dynamic dispatch, column-major
inner loops, kernels that skip the structural zeros of the à trous filters, and
the preallocated-buffer workspace. GPU execution is not in the table because the
MATLAB toolboxes have no GPU path to compare against; see
[Performance, threading and GPUs](@ref performance).

### Reproducing

Clone the two toolboxes into `matlab/` and compile their MEX kernels as
described in [`matlab/README.md`](https://github.com/hakkelt/Contourlets.jl/blob/master/matlab/README.md),
then from MATLAB:

```matlab
verify_vs_julia(2, [2 3], [64 128 256])   % J, L_array, sizes
```

The script writes shared inputs, runs the Julia side as a subprocess
(`JULIA_BIN` selects the binary), and reports PR errors, cross-validation
errors and timings for both. For Julia-only benchmarks use `benchmark/benchmarks.jl`.

---

## Implementation notes

### Differences from the MATLAB CT toolbox

1. **Filter convention**: The MATLAB toolbox names filters by string — `'9-7'`
   for the Laplacian-Pyramid CDF 9/7 pair (exposed here as [`CDF97`](@ref)) and
   `'pkva'` for the directional ladder filter (exposed here as [`Q2345`](@ref)).
   These are two *different* filters for two different stages, not aliases.

2. **Subband ordering**: The MATLAB toolbox returns subbands as a cell array ordered
   by the binary tree traversal.  Contourlets.jl uses the same ordering (`subbands[j][k]`
   where `k` runs left-to-right in the binary tree).

3. **Boundary conditions**: Both use symmetric (reflect) boundary extension by default.
   Contourlets.jl additionally supports `:periodic` and `:zero` via the `boundary` keyword.

4. **Workspace API**: The MATLAB toolbox allocates on every call.  Contourlets.jl
   provides `ct_forward!(coeffs, img, params; workspace=ws)` which reuses
   preallocated buffers for the pyramid stage (the directional stage still
   allocates its subband tree when no workspace is supplied).

### Differences from the MATLAB NSCT toolbox

1. **À trous implementation**: Both use zero-insertion filter upsampling
   (`upsample_filter`) for the à trous scheme.

2. **NSDFB tree level**: The `tree_level` argument in `nsdfb_decompose` corresponds to the
   recursion depth in the MATLAB toolbox's internal `nsdfbdec` helper.

3. **Synthesis**: Contourlets.jl realises the directional stage with the Phoong
   ladder network, whose synthesis is the structural inverse of the analysis
   ladder (exact PR by construction).  All NSP/NSDFB filtering is periodic, so
   the transform is exactly invariant under circular shifts of the input.
