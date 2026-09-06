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
    GPUEnv.activate(persist = true, include_jlarrays = false, only_first = true)
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
SUITE["LP"] = BenchmarkGroup()
SUITE["DFB"] = BenchmarkGroup()
SUITE["NSDFB"] = BenchmarkGroup()
SUITE["primitives"] = BenchmarkGroup()
SUITE["NSCT3"] = BenchmarkGroup()
SUITE["NSDFB3"] = BenchmarkGroup()
SUITE["primitives3"] = BenchmarkGroup()

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

        # Setup CT with J=3, L_array=[2, 3, 4]
        p_ct = ContourletParams(J = 3, L_array = [2, 3, 4])
        ws_ct = make_workspace(T, (sz, sz), p_ct)
        coeffs_ct_alloc = similar_coefficients(p_ct, (sz, sz); Td = T)

        # CT Forward: allocating version
        SUITE["CT"][sz_str]["forward"] = @benchmarkable ct_forward($img, $p_ct)

        # CT Forward: in-place with workspace
        SUITE["CT"][sz_str]["forward!"] = @benchmarkable ct_forward!($coeffs_ct_alloc, $img, $p_ct; workspace = $ws_ct)

        # Get coefficients for inverse benchmarks
        coeffs_ct = ct_forward(img, p_ct)

        # CT Inverse: allocating version
        SUITE["CT"][sz_str]["inverse"] = @benchmarkable ct_inverse($coeffs_ct, $p_ct)

        # CT Inverse: in-place with workspace
        img_out_ct = similar(img)
        SUITE["CT"][sz_str]["inverse!"] = @benchmarkable ct_inverse!($img_out_ct, $coeffs_ct, $p_ct; workspace = $ws_ct)

        # Setup NSCT with J=3, L_array=[2, 3, 4] to highlight the FFT cascade speedup
        p_nsct = ContourletParams(J = 3, L_array = [2, 3, 4])
        ws_nsct = make_nsct_workspace(T, (sz, sz), p_nsct)
        coeffs_nsct_alloc = similar_nsct_coefficients(p_nsct, (sz, sz); Td = T)

        # NSCT Forward: allocating version
        SUITE["NSCT"][sz_str]["forward"] = @benchmarkable nsct_forward($img, $p_nsct)

        # NSCT Forward: in-place with workspace
        SUITE["NSCT"][sz_str]["forward!"] = @benchmarkable nsct_forward!($coeffs_nsct_alloc, $img, $p_nsct; workspace = $ws_nsct)

        # Get NSCT coefficients for inverse benchmark
        coeffs_nsct = nsct_forward(img, p_nsct)

        # NSCT Inverse: allocating version
        SUITE["NSCT"][sz_str]["inverse"] = @benchmarkable nsct_inverse($coeffs_nsct, $p_nsct)

        # NSCT Inverse: in-place with workspace
        img_out_nsct = similar(img)
        SUITE["NSCT"][sz_str]["inverse!"] = @benchmarkable nsct_inverse!($img_out_nsct, $coeffs_nsct, $p_nsct; workspace = $ws_nsct)

        # Optional GPU Benchmarks if a backend is available
        if @isdefined(GPU_BACKEND)
            img_gpu = to_gpu(GPU_BACKEND, img)
            Tf = real(T)
            p_ct_gpu = ContourletParams(
                J = 3, L_array = [2, 3, 4],
                lp_filters = FilterPair{Tf}(Tf.(CDF97.h), Tf.(CDF97.g)),
                dfb_filters = QuincunxFilterPair{Tf}(
                    Tf.(Q2345.h_q), Tf.(Q2345.g_q),
                    Q2345.c_h, Q2345.c_g, Tf.(Q2345.f_ladder)
                )
            )
            ws_ct_gpu = make_workspace(img_gpu, p_ct_gpu)
            coeffs_ct_gpu_alloc = similar_coefficients(p_ct_gpu, (sz, sz); Td = T, M = typeof(img_gpu))

            SUITE["CT"][sz_str]["forward_gpu"] = @benchmarkable begin
                ct_forward($img_gpu, $p_ct_gpu)
                synchronize_backend($GPU_BACKEND)
            end
            SUITE["CT"][sz_str]["forward!_gpu"] = @benchmarkable begin
                ct_forward!($coeffs_ct_gpu_alloc, $img_gpu, $p_ct_gpu; workspace = $ws_ct_gpu)
                synchronize_backend($GPU_BACKEND)
            end

            coeffs_ct_gpu = ct_forward(img_gpu, p_ct_gpu)
            SUITE["CT"][sz_str]["inverse_gpu"] = @benchmarkable begin
                ct_inverse($coeffs_ct_gpu, $p_ct_gpu)
                synchronize_backend($GPU_BACKEND)
            end

            img_out_ct_gpu = similar(img_gpu)
            SUITE["CT"][sz_str]["inverse!_gpu"] = @benchmarkable begin
                ct_inverse!($img_out_ct_gpu, $coeffs_ct_gpu, $p_ct_gpu; workspace = $ws_ct_gpu)
                synchronize_backend($GPU_BACKEND)
            end

            p_nsct_gpu = ContourletParams(
                J = 3, L_array = [2, 3, 4],
                lp_filters = FilterPair{Tf}(Tf.(CDF97.h), Tf.(CDF97.g)),
                dfb_filters = QuincunxFilterPair{Tf}(
                    Tf.(Q2345.h_q), Tf.(Q2345.g_q),
                    Q2345.c_h, Q2345.c_g, Tf.(Q2345.f_ladder)
                )
            )
            ws_nsct_gpu = make_nsct_workspace(img_gpu, p_nsct_gpu)
            coeffs_nsct_gpu_alloc = similar_nsct_coefficients(p_nsct_gpu, (sz, sz); Td = T, M = typeof(img_gpu))

            SUITE["NSCT"][sz_str]["forward_gpu"] = @benchmarkable begin
                nsct_forward($img_gpu, $p_nsct_gpu)
                synchronize_backend($GPU_BACKEND)
            end
            SUITE["NSCT"][sz_str]["forward!_gpu"] = @benchmarkable begin
                nsct_forward!($coeffs_nsct_gpu_alloc, $img_gpu, $p_nsct_gpu; workspace = $ws_nsct_gpu)
                synchronize_backend($GPU_BACKEND)
            end

            coeffs_nsct_gpu = nsct_forward(img_gpu, p_nsct_gpu)
            SUITE["NSCT"][sz_str]["inverse_gpu"] = @benchmarkable begin
                nsct_inverse($coeffs_nsct_gpu, $p_nsct_gpu)
                synchronize_backend($GPU_BACKEND)
            end

            img_out_nsct_gpu = similar(img_gpu)
            SUITE["NSCT"][sz_str]["inverse!_gpu"] = @benchmarkable begin
                nsct_inverse!($img_out_nsct_gpu, $coeffs_nsct_gpu, $p_nsct_gpu; workspace = $ws_nsct_gpu)
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

                # DFB Forward: dfb_decompose(image::Matrix, level::Int, qfp::QuincunxFilterPair) → Vector{Matrix}
                SUITE["DFB"][sz_str]["$(level_str) forward"] =
                    @benchmarkable dfb_decompose($img, $level, Q2345)

                # Get subbands for reconstruction benchmark
                subbands_dfb = dfb_decompose(img, level, Q2345)

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
                    @benchmarkable nsdfb_decompose($img, $level, Q2345, 1)
                subbands_ns = nsdfb_decompose(img, level, Q2345, 1)
                SUITE["NSDFB"][sz_str]["$(level_str) inverse"] =
                    @benchmarkable nsdfb_reconstruct($subbands_ns, Q2345, 1)
            end

            # ────────────────────────────────────────────────────────────────────────────
            # Low-level primitives
            # ────────────────────────────────────────────────────────────────────────────

            SUITE["primitives"][sz_str] = BenchmarkGroup()
            # Separable convolution (LP workhorse) and the direct/FFTW conv2d backends.
            k_small = Float64[1.0 2.0 1.0]                       # direct backend (≤25 taps)
            k_large = randn(RNG, Float64, 7, 7)                  # FFTW backend (>25 taps)
            SUITE["primitives"][sz_str]["conv2d_sep"] =
                @benchmarkable conv2d_sep($img, [1.0, 2.0, 1.0], [1.0, 2.0, 1.0])
            SUITE["primitives"][sz_str]["conv2d_direct"] =
                @benchmarkable conv2d($img, $k_small, (1, 2))
            SUITE["primitives"][sz_str]["conv2d_fftw"] =
                @benchmarkable conv2d($img, $k_large, (4, 4))
        end
    end
end

# ╔════════════════════════════════════════════════════════════════════════════╗
# ║ 3-D SUITE                                                                  ║
# ╚════════════════════════════════════════════════════════════════════════════╝
#
# Volume sizes are 32³ and 64³, not the 256³ working point: this suite runs once
# per compared revision in CI, and a 256³ NSCT3 forward pass alone is seconds.
# Plans are built with FFTW.ESTIMATE so planning does not dominate the setup.
#
# Float32 only — the 3-D path treats a complex volume as two real passes through
# the same kernel, so there is no separate threading regime to measure.

for n in (32, 64)
    sz_str = "$(n)^3"
    vol = randn(RNG, Float32, n, n, n)

    # ── NSCT3 ────────────────────────────────────────────────────────────────
    SUITE["NSCT3"][sz_str] = BenchmarkGroup()
    p3 = ContourletParams3(J = 2, L_array = [1, 2], order = 12)
    ws3 = make_nsct3_workspace(p3, (n, n, n), Float32; flags = FFTW.ESTIMATE)
    SUITE["NSCT3"][sz_str]["forward"] = @benchmarkable nsct3_forward($vol, $p3)
    SUITE["NSCT3"][sz_str]["forward (workspace)"] =
        @benchmarkable nsct3_forward($vol, $p3; workspace = $ws3)
    # The streaming form is the one that matters at working sizes: it holds one
    # subband instead of `1 + Σ 3·2^L` volumes.
    SUITE["NSCT3"][sz_str]["forward (streaming)"] =
        @benchmarkable nsct3_forward((j, k, b) -> nothing, $vol, $p3; workspace = $ws3)
    coeffs3 = nsct3_forward(vol, p3; workspace = ws3)
    out3 = similar(vol)
    SUITE["NSCT3"][sz_str]["inverse"] = @benchmarkable nsct3_inverse($coeffs3, $p3)
    SUITE["NSCT3"][sz_str]["inverse!"] =
        @benchmarkable nsct3_inverse!($out3, $coeffs3, $p3; workspace = $ws3)

    # ── NSDFB3 ───────────────────────────────────────────────────────────────
    SUITE["NSDFB3"][sz_str] = BenchmarkGroup()
    for L in (1, 3)
        fb3 = make_nsdfb3((n, n, n), Float32; levels = L, order = 12, flags = FFTW.ESTIMATE)
        SUITE["NSDFB3"][sz_str]["L=$(L) forward (streaming)"] =
            @benchmarkable nsdfb3_decompose((k, b) -> nothing, $vol, $fb3)
    end
    hg3 = make_hourglass3((n, n, n), Float32; order = 12, flags = FFTW.ESTIMATE)
    dst3 = ntuple(_ -> similar(vol), 3)
    SUITE["NSDFB3"][sz_str]["hourglass forward"] =
        @benchmarkable hourglass3_decompose!($dst3, $vol, $hg3)

    # ── 3-D primitives ───────────────────────────────────────────────────────
    SUITE["primitives3"][sz_str] = BenchmarkGroup()
    h_short = Float32[0.25, 0.5, 0.25]
    h_long = upsample_filter(Float32.(CDF97.h), 4)      # 33 taps: the FFT regime
    dst = similar(vol)
    tmp = similar(vol)
    fftplan = make_conv3d_fft_plan((n, n, n), Float32; flags = FFTW.ESTIMATE)
    SUITE["primitives3"][sz_str]["conv3d_sep direct (3 taps)"] =
        @benchmarkable conv3d_sep!($dst, $vol, $h_short, $h_short, $h_short; tmp = $tmp, backend = :direct)
    SUITE["primitives3"][sz_str]["conv3d_sep direct (33 taps)"] =
        @benchmarkable conv3d_sep!(
        $dst, $vol, $h_long, $h_long, $h_long;
        boundary = :periodic, tmp = $tmp, backend = :direct
    )
    SUITE["primitives3"][sz_str]["conv3d_sep fft (33 taps)"] =
        @benchmarkable conv3d_sep_fft!($dst, $vol, $h_long, $h_long, $h_long, $fftplan)

    coarse3 = similar(vol)
    bp3 = similar(vol)
    tmp23 = similar(vol)
    SUITE["primitives3"][sz_str]["nsp3_decompose!"] = @benchmarkable nsp3_decompose!(
        $coarse3, $bp3, $vol, CDF97, 3; tmp = $tmp, tmp2 = $tmp23, plan = $fftplan
    )
end
