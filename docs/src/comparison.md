# [Comparison with the MATLAB toolboxes](@id comparison)

This page compares Contourlets.jl against the two reference implementations:

- **MATLAB Contourlet Toolbox** (Do & Vetterli, 2005) — [FileExchange 8837](https://www.mathworks.com/matlabcentral/fileexchange/8837-contourlet-toolbox), GNU GPL (per its `COPYING`)
- **MATLAB Nonsubsampled Contourlet Toolbox** (da Cunha et al., 2006) — [FileExchange 10049](https://www.mathworks.com/matlabcentral/fileexchange/10049-nonsubsampled-contourlet-toolbox), no licence stated

Neither toolbox is included in this MIT-licensed repository, and no code here is
derived from them: the package implements the papers (see [Theory](@ref theory)),
and the toolboxes are run only as black boxes, to compare outputs and timings.
Clone them into `matlab/` as described in
[`matlab/README.md`](https://github.com/hakkelt/Contourlets.jl/blob/master/matlab/README.md)
to reproduce the validation and benchmarks below.

---

## Feature comparison

The transforms, against the two MATLAB toolboxes.  The volume transforms live in
[Surfacelets.jl](https://github.com/hakkelt/Surfacelets.jl) and are compared
against SurfBox there.

| Feature | Contourlets.jl | MATLAB CT Toolbox | MATLAB NSCT Toolbox |
|---------|---------------|------------------|-------------------|
| Language | Julia 1.10+ | MATLAB R2006+ | MATLAB R2007+ |
| Licence | MIT | GPL | none stated |
| Discrete CT (`ct_forward/inverse`) | ✅ | ✅ (`pdfbdec/pdfbrec`) | ✅ |
| Nonsubsampled CT (`nsct_forward/inverse`) | ✅ | ❌ | ✅ (`nsctdec/nsctrec`) |
| Translation-invariant CT, TICT / STICT (Eslami & Radha 2006) | ✅ | ❌ | ❌ |
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

The DFB uses the same filters as the toolbox and agrees with its `dfbdec_l` to
1e-15 for `L ≤ 2` (a regression test locks this in).  From `L = 3` the two
differ.  This package computes the multichannel bank of Do's thesis
(eq. 3.23), whose subbands commute with input shifts on their own sampling
lattices; the toolbox's subbands do not, not even for an `(8, 8)` shift that lies
on every lattice.  The NSDFB here is the DFB's equivalent filters without
decimation, with the decimation delays dropped (da Cunha et al. 2006).  Its
subbands match the DFB's subband for subband.  They differ from `nsdfbdec`'s by
a shift and a sign for `L ≤ 2`, and in the wedges themselves from `L = 3`.
`nsdfbdec` does not resolve every wedge from `L = 4` on: in a plane wave sweeping
180°, 8 of its 16 subbands (16 of 32 at `L = 5`) never carry the most energy,
while here every subband does, except the two at the fan boundary at `L = 5`
(see [Theory](@ref theory)).  The CT
and NSCT also use different pyramid filters by design: this package uses the
JPEG 2000-normalised separable CDF 9/7 à trous pyramid, the toolboxes a
differently scaled `'9-7'` and a non-separable diamond pyramid.  So only PR is
comparable there.

Perfect reconstruction in the directional stage is *structural*.  The quincunx
pair of Phoong et al. (1995) satisfies the perfect-reconstruction conditions for
any ladder filter, so the equivalent filters satisfy ``\sum_k G_k H_k = 2^L``
identically.  That is why the residuals are at the floating-point floor rather
than merely small.

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

3. **Boundary conditions**: The Laplacian pyramid uses symmetric (reflect)
   extension by default (`:periodic` and `:zero` are available through the
   `boundary` keyword).  The DFB is periodic: it filters on the DFT grid.

4. **DFB from `L = 3`**: the toolbox's deeper tree is not shift-equivariant on
   its sampling lattices, while this package computes the thesis' multichannel
   bank exactly, so the coefficients differ (see Numerical accuracy above).

5. **Workspace API**: The MATLAB toolbox allocates on every call.  Contourlets.jl
   provides `ct_forward!(coeffs, img, params; workspace=ws)`, which reuses the
   pyramid buffers and the directional stage's precomputed filters, FFT buffers
   and plans, allocating nothing per call.

### Differences from the MATLAB NSCT toolbox

1. **À trous implementation**: Both use zero-insertion filter upsampling
   (`upsample_filter`) for the à trous scheme.

2. **Coarse-scale directional filters**: at pyramid level `j ≥ 3` the NSDFB
   filters are upsampled by ``2^{j-2} I`` (the `tree_level` argument of
   `nsdfb_decompose`), following §II-A.3 of the NSCT paper, which leaves the
   choice of factor open; this one was chosen by measuring directional leakage
   (see [Theory](@ref theory)).

3. **NSDFB construction**: Contourlets.jl filters with the DFB's equivalent
   filters at full rate (the NSCT paper's "DFB with the samplers switched off"),
   zero-phase, so every subband is registered with the input.  The filters
   satisfy ``\sum_k G_k H_k = 1`` exactly.  All NSP/NSDFB filtering is periodic, so
   the transform is exactly invariant under circular shifts of the input.
