"""
    benchmarks.jl — Contourlets.jl BenchmarkTools SUITE

USAGE:
    julia --project=benchmark benchmark/benchmarks.jl

This file defines a comprehensive benchmark suite for the Contourlets package
covering all major transforms:
  - Contourlet Transform (CT) — forward/inverse, allocating and in-place
  - Non-Subsampled Contourlet Transform (NSCT) — forward/inverse  
  - Laplacian Pyramid (LP) — decomposition and reconstruction
  - Directional Filter Bank (DFB) — decomposition and reconstruction at multiple levels

Benchmarks are run over two image sizes: 64×64, 256×256. LP/DFB/NSDFB/primitives
run only for Float64 (CT/NSCT cover the Float64-vs-ComplexF64 threading split).
"""

using BenchmarkTools, Contourlets, Random, GPUEnv
using Contourlets: lp_decompose, lp_reconstruct, dfb_decompose, dfb_reconstruct,
    nsdfb_decompose, nsdfb_reconstruct, conv2d_sep
using FFTW: FFTW

# Cap each @benchmarkable's tuning + steady-state phase: AirspeedVelocity runs this
# suite once per compared revision in CI, and revision-to-revision comparisons don't
# need BenchmarkTools' full-precision default budget.
BenchmarkTools.DEFAULT_PARAMETERS.samples = 3
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 1.0

# Detect if a GPU is available and activate it
const HARDWARE = GPUEnv.detect_gpu_hardware()
const GPU_DETECTED = any(k -> get(HARDWARE, k, false) == 1, (:CUDA, :AMDGPU, :Metal, :oneAPI))

if GPU_DETECTED
    GPUEnv.activate(persist = true, include_cpu_backends = false, only_first = true)
    const BACKENDS = gpu_backends()
    if !isempty(BACKENDS)
        const GPU_BACKEND = first(BACKENDS)
    end
end

# ╔════════════════════════════════════════════════════════════════════════════╗
# ║ SUITE DEFINITION                                                           ║
# ╚════════════════════════════════════════════════════════════════════════════╝

const SUITE = BenchmarkGroup()
SUITE["CT"] = BenchmarkGroup()
SUITE["NSCT"] = BenchmarkGroup()
SUITE["TICT"] = BenchmarkGroup()
SUITE["STICT"] = BenchmarkGroup()
SUITE["LP"] = BenchmarkGroup()
SUITE["DFB"] = BenchmarkGroup()
SUITE["NSDFB"] = BenchmarkGroup()
SUITE["primitives"] = BenchmarkGroup()

# ╔════════════════════════════════════════════════════════════════════════════╗
# ║ SETUP: Random number generator & test sizes                                ║
# ╚════════════════════════════════════════════════════════════════════════════╝

const RNG = MersenneTwister(1234)
# Small + large image sizes bracket the CT/NSCT complexity trend.
const TEST_SIZES = [64, 256]

# ╔════════════════════════════════════════════════════════════════════════════╗
# ║ SIZE-SPECIFIC BENCHMARKS                                                   ║
# ╚════════════════════════════════════════════════════════════════════════════╝

for sz in TEST_SIZES
    for T in [Float64, ComplexF64]
        sz_str = "$(sz)x$(sz)_$(T)"

        # Generate test image for this size and type
        img = randn(RNG, T, sz, sz)

        # ────────────────────────────────────────────────────────────────────────────
        # Contourlet Transform (CT): forward/inverse
        # ────────────────────────────────────────────────────────────────────────────

        SUITE["CT"][sz_str] = BenchmarkGroup()

        # Setup CT with scales = 3, direction_levels = [2, 3, 4]
        t_ct = CT(scales = 3, direction_levels = [2, 3, 4])
        ws_ct = make_workspace(t_ct, img)
        coeffs_ct_alloc = similar_coefficients(t_ct, img)

        # CT Forward: allocating version
        SUITE["CT"][sz_str]["forward"] = @benchmarkable analyze($t_ct, $img)

        # CT Forward: in-place with workspace
        SUITE["CT"][sz_str]["forward!"] = @benchmarkable analyze!($coeffs_ct_alloc, $t_ct, $img; workspace = $ws_ct)

        # Get coefficients for inverse benchmarks
        coeffs_ct = analyze(t_ct, img)

        # CT Inverse: allocating version
        SUITE["CT"][sz_str]["inverse"] = @benchmarkable synthesize($t_ct, $coeffs_ct)

        # CT Inverse: in-place with workspace
        img_out_ct = similar(img)
        SUITE["CT"][sz_str]["inverse!"] = @benchmarkable synthesize!($img_out_ct, $t_ct, $coeffs_ct; workspace = $ws_ct)

        # Setup NSCT with scales = 3, direction_levels = [2, 3, 4] to highlight the FFT cascade speedup
        t_nsct = NSCT(scales = 3, direction_levels = [2, 3, 4])
        ws_nsct = make_workspace(t_nsct, img)
        coeffs_nsct_alloc = similar_coefficients(t_nsct, img)

        # NSCT Forward: allocating version
        SUITE["NSCT"][sz_str]["forward"] = @benchmarkable analyze($t_nsct, $img)

        # NSCT Forward: in-place with workspace
        SUITE["NSCT"][sz_str]["forward!"] = @benchmarkable analyze!($coeffs_nsct_alloc, $t_nsct, $img; workspace = $ws_nsct)

        # Get NSCT coefficients for inverse benchmark
        coeffs_nsct = analyze(t_nsct, img)

        # NSCT Inverse: allocating version
        SUITE["NSCT"][sz_str]["inverse"] = @benchmarkable synthesize($t_nsct, $coeffs_nsct)

        # NSCT Inverse: in-place with workspace
        img_out_nsct = similar(img)
        SUITE["NSCT"][sz_str]["inverse!"] = @benchmarkable synthesize!($img_out_nsct, $t_nsct, $coeffs_nsct; workspace = $ws_nsct)

        # Subband FFTs split between threads (one task per Julia thread)
        SUITE["NSCT"][sz_str]["forward! threaded"] = @benchmarkable analyze!($coeffs_nsct_alloc, $t_nsct, $img; workspace = $ws_nsct, threaded = true)
        SUITE["NSCT"][sz_str]["inverse! threaded"] = @benchmarkable synthesize!($img_out_nsct, $t_nsct, $coeffs_nsct; workspace = $ws_nsct, threaded = true)

        # TICT / STICT (Eslami & Radha): workspace paths, same parameters as the NSCT
        for (name, t_ti) in (("TICT", TICT(scales = 3, direction_levels = [2, 3, 4])), ("STICT", STICT(scales = 3, direction_levels = [2, 3, 4])))
            ws_ti = make_workspace(t_ti, img)
            coeffs_ti = similar_coefficients(t_ti, img)
            img_out_ti = similar(img)
            SUITE[name][sz_str]["forward!"] = @benchmarkable analyze!($coeffs_ti, $t_ti, $img; workspace = $ws_ti)
            SUITE[name][sz_str]["inverse!"] = @benchmarkable synthesize!($img_out_ti, $t_ti, $coeffs_ti; workspace = $ws_ti)
        end
        t_tt = TICT(scales = 3, direction_levels = [2, 3, 4])
        ws_tt = make_workspace(t_tt, img)
        coeffs_tt = similar_coefficients(t_tt, img)
        img_out_tt = similar(img)
        SUITE["TICT"][sz_str]["forward! threaded"] = @benchmarkable analyze!($coeffs_tt, $t_tt, $img; workspace = $ws_tt, threaded = true)
        SUITE["TICT"][sz_str]["inverse! threaded"] = @benchmarkable synthesize!($img_out_tt, $t_tt, $coeffs_tt; workspace = $ws_tt, threaded = true)

        # Optional GPU Benchmarks if a backend is available
        if @isdefined(GPU_BACKEND)
            img_gpu = to_gpu(GPU_BACKEND, img)
            ws_ct_gpu = make_workspace(t_ct, img_gpu)
            coeffs_ct_gpu_alloc = similar_coefficients(t_ct, img_gpu)

            SUITE["CT"][sz_str]["forward_gpu"] = @benchmarkable begin
                analyze($t_ct, $img_gpu)
                synchronize_backend($GPU_BACKEND)
            end
            SUITE["CT"][sz_str]["forward!_gpu"] = @benchmarkable begin
                analyze!($coeffs_ct_gpu_alloc, $t_ct, $img_gpu; workspace = $ws_ct_gpu)
                synchronize_backend($GPU_BACKEND)
            end

            coeffs_ct_gpu = analyze(t_ct, img_gpu)
            SUITE["CT"][sz_str]["inverse_gpu"] = @benchmarkable begin
                synthesize($t_ct, $coeffs_ct_gpu)
                synchronize_backend($GPU_BACKEND)
            end

            img_out_ct_gpu = similar(img_gpu)
            SUITE["CT"][sz_str]["inverse!_gpu"] = @benchmarkable begin
                synthesize!($img_out_ct_gpu, $t_ct, $coeffs_ct_gpu; workspace = $ws_ct_gpu)
                synchronize_backend($GPU_BACKEND)
            end

            ws_nsct_gpu = make_workspace(t_nsct, img_gpu)
            coeffs_nsct_gpu_alloc = similar_coefficients(t_nsct, img_gpu)

            SUITE["NSCT"][sz_str]["forward_gpu"] = @benchmarkable begin
                analyze($t_nsct, $img_gpu)
                synchronize_backend($GPU_BACKEND)
            end
            SUITE["NSCT"][sz_str]["forward!_gpu"] = @benchmarkable begin
                analyze!($coeffs_nsct_gpu_alloc, $t_nsct, $img_gpu; workspace = $ws_nsct_gpu)
                synchronize_backend($GPU_BACKEND)
            end

            coeffs_nsct_gpu = analyze(t_nsct, img_gpu)
            SUITE["NSCT"][sz_str]["inverse_gpu"] = @benchmarkable begin
                synthesize($t_nsct, $coeffs_nsct_gpu)
                synchronize_backend($GPU_BACKEND)
            end

            img_out_nsct_gpu = similar(img_gpu)
            SUITE["NSCT"][sz_str]["inverse!_gpu"] = @benchmarkable begin
                synthesize!($img_out_nsct_gpu, $t_nsct, $coeffs_nsct_gpu; workspace = $ws_nsct_gpu)
                synchronize_backend($GPU_BACKEND)
            end
        end

        # LP/DFB/NSDFB/primitives run Float64 only: CT/NSCT above dispatch through
        # this same pyramid/DFB/NSDFB code, so they already cover the real-vs-complex
        # threading-path split that motivates sweeping T.
        if T === Float64
            # ────────────────────────────────────────────────────────────────────────────
            # Laplacian Pyramid (LP): decomposition and reconstruction
            # ────────────────────────────────────────────────────────────────────────────

            SUITE["LP"][sz_str] = BenchmarkGroup()

            # LP Decomposition
            SUITE["LP"][sz_str]["decompose"] = @benchmarkable lp_decompose($img, CDF97)

            # Get LP coefficients for reconstruction benchmark
            coarse_lp, bandpass_lp = lp_decompose(img, CDF97)

            # LP Reconstruction: lp_reconstruct(coarse::Matrix, bandpass::Matrix, fp::FilterPair) → Matrix
            SUITE["LP"][sz_str]["reconstruct"] = @benchmarkable lp_reconstruct($coarse_lp, $bandpass_lp, CDF97)

            # ────────────────────────────────────────────────────────────────────────────
            # Directional Filter Bank (DFB): decomposition and reconstruction
            # ────────────────────────────────────────────────────────────────────────────

            SUITE["DFB"][sz_str] = BenchmarkGroup()

            # DFB at the shallowest and deepest decomposition levels (1, 4) — the
            # intermediate levels (2, 3) interpolate between these two data points.
            for level in [1, 4]
                level_str = "L=$level"

                # DFB Forward: dfb_decompose(image::Matrix, qfp::QuincunxFilterPair, level::Int) → Vector{Matrix}
                SUITE["DFB"][sz_str]["$(level_str) forward"] =
                    @benchmarkable dfb_decompose($img, Q2345, $level)

                # Get subbands for reconstruction benchmark
                subbands_dfb = dfb_decompose(img, Q2345, level)

                # DFB Inverse: dfb_reconstruct(subbands::Vector{Matrix}, qfp::QuincunxFilterPair) → Matrix
                SUITE["DFB"][sz_str]["$(level_str) inverse"] =
                    @benchmarkable dfb_reconstruct($subbands_dfb, Q2345)
            end

            # ────────────────────────────────────────────────────────────────────────────
            # Non-Subsampled Directional Filter Bank (NSDFB)
            # ────────────────────────────────────────────────────────────────────────────

            SUITE["NSDFB"][sz_str] = BenchmarkGroup()
            for level in [1, 3]
                level_str = "L=$level"
                SUITE["NSDFB"][sz_str]["$(level_str) forward"] =
                    @benchmarkable nsdfb_decompose($img, Q2345, $level; tree_level = 1)
                subbands_ns = nsdfb_decompose(img, Q2345, level; tree_level = 1)
                SUITE["NSDFB"][sz_str]["$(level_str) inverse"] =
                    @benchmarkable nsdfb_reconstruct($subbands_ns, Q2345; tree_level = 1)
            end

            # ────────────────────────────────────────────────────────────────────────────
            # Low-level primitives
            # ────────────────────────────────────────────────────────────────────────────

            SUITE["primitives"][sz_str] = BenchmarkGroup()
            # Separable convolution, the Laplacian pyramid's workhorse.
            SUITE["primitives"][sz_str]["conv2d_sep"] =
                @benchmarkable conv2d_sep($img, [1.0, 2.0, 1.0], [1.0, 2.0, 1.0])
        end
    end
end
