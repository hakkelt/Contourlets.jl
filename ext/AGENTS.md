# GPU extension notes

- **GPU = whole transform on the device** — the `ContourletsGPUExt` extension
  runs every stage of CT/NSCT/TICT/STICT on the device. Pyramid functions are
  reused unchanged (broadcast-based dispatch picks up GPU primitives
  automatically); the directional banks and TILP masks only need moving to the
  device (`_adapt_bank`, `_like`) and an FFT for the array type (cuFFT, rocFFT;
  JLArrays through FFTW on host memory — not OpenCL). JLArrays' `rfft` returns
  a `Matrix`, so allocate outputs with `similar` and fill them via plans + `mul!`.
  Coefficients stay device-resident (`ContourletCoefficients{Td,A}`,
  `A<:AbstractMatrix`); use `Array(·)` / `Adapt.adapt(Array, coeffs)` to bring them
  back. `ContourletsCUDAExt` adds CUDA graph capture for the workspace paths. Load
  the `my-gpu-ext` skill before touching any file under `ext/`.
