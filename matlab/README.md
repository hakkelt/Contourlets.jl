# MATLAB reference implementations

This folder vendors two upstream MATLAB toolboxes as **git submodules**. They are
the canonical references for the (nonsubsampled) contourlet transform and are used
only to validate and benchmark `Contourlets.jl` — they are not part of the Julia
package and are not required to use it.

| Submodule | Upstream | What it provides |
|---|---|---|
| `Contourlet-transform/` | [aliibraheemshash/Contourlet-transform](https://github.com/aliibraheemshash/Contourlet-transform) (`master` branch) | Do & Vetterli critically-sampled Contourlet Toolbox (`pdfbdec`, `dfbdec`, `dfilters`, …). |
| `Nonsubsampled-Contourlet-Toolbox/nsct_toolbox/` | [ssalili/Nonsubsampled-Contourlet-Toolbox](https://github.com/ssalili/Nonsubsampled-Contourlet-Toolbox) | da Cunha–Zhou–Do Nonsubsampled Contourlet Toolbox (`nsctdec`, `nsdfbdec`, `nssfbdec`, `parafilters`, …). |

The NSCT toolbox is the reference for the resampling-matrix NSDFB construction
(fan + parallelogram filters convolved with upsampling matrices), which is the
shift-invariant directional filter bank `Contourlets.jl` implements.

## Getting the sources

```bash
git submodule update --init --recursive
```

> The CT repo keeps the actual toolbox on its `master` branch (its `main` branch is
> only a README); the submodule is pinned to `master` accordingly.

## Building the MEX files

Both toolboxes ship C sources that must be compiled once before use (Linux builds
`*.mexa64`). From MATLAB, in each toolbox directory:

```matlab
mex resampc.c                      % Contourlet-transform/
mex atrousc.c zconv2.c zconv2S.c   % Nonsubsampled-Contourlet-Toolbox/nsct_toolbox/
```

On this system MATLAB is provided as a module — run `module load matlab` once per
shell before invoking `matlab`.

## Cross-validation

`verify_vs_julia.m` (this folder) runs the MATLAB reference and the Julia
implementation on identical inputs, checks they agree numerically, and benchmarks
both. See its header for usage.
