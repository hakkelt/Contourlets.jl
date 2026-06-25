# MATLAB reference implementations

This folder references two upstream MATLAB toolboxes used only to validate and
benchmark `Contourlets.jl` — they are not part of the Julia package and are not
required to use it.

| Directory | Upstream | What it provides |
|---|---|---|
| `Contourlet-transform/` | [aliibraheemshash/Contourlet-transform](https://github.com/aliibraheemshash/Contourlet-transform) (`master` branch) | Do & Vetterli critically-sampled Contourlet Toolbox (`pdfbdec`, `dfbdec`, `dfilters`, …). |
| `Nonsubsampled-Contourlet-Toolbox/` | [ssalili/Nonsubsampled-Contourlet-Toolbox](https://github.com/ssalili/Nonsubsampled-Contourlet-Toolbox) | da Cunha–Zhou–Do Nonsubsampled Contourlet Toolbox (`nsctdec`, `nsdfbdec`, `nssfbdec`, `parafilters`, …). |

The NSCT toolbox is the reference for the resampling-matrix NSDFB construction
(fan + parallelogram filters convolved with upsampling matrices), which is the
shift-invariant directional filter bank `Contourlets.jl` implements.

## Getting the sources

Clone each toolbox directly into the expected directory:

```bash
git clone https://github.com/aliibraheemshash/Contourlet-transform.git \
    --branch master --single-branch matlab/Contourlet-transform

git clone https://github.com/ssalili/Nonsubsampled-Contourlet-Toolbox.git \
    --single-branch matlab/Nonsubsampled-Contourlet-Toolbox
```

> The CT repo keeps the actual toolbox on its `master` branch (its `main` branch is
> only a README); clone accordingly.
>
> Do **not** clone inside the repo with `git init` in the destination — plain
> directories work fine and avoid interfering with the parent git repository.

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
