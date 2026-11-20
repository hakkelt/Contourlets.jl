# GPU extension notes

- **GPU = whole transform on the device** — the `ContourletsGPUExt` extension
  runs every stage of CT/NSCT/TICT/STICT on the device. Pyramid functions are
  reused unchanged (broadcast-based dispatch picks up GPU primitives
  automatically); the directional banks and TILP masks only need moving to the
  device (`_adapt_bank`, `_like`) and an FFT for the array type (cuFFT, rocFFT;
  JLArrays through FFTW on host memory — not OpenCL). JLArrays' `rfft` returns
  a `Matrix`, so allocate outputs with `similar` and fill them via plans + `mul!`.
  Coefficients are allocated like the input (a device stacked array or a
  `SubbandVector` over a device vector), so they stay device-resident; use
  `Adapt.adapt(Array, c)` (or `Array(parent(c))` for a `SubbandVector`) to bring
  them back. `ContourletsCUDAExt` adds CUDA graph capture for the workspace paths. Load
  the `my-gpu-ext` skill before touching any file under `ext/`.
