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

Benchmarks are run over three image sizes: 64×64, 128×128, 256×256.
"""

using BenchmarkTools, Contourlets, Random

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

# ╔════════════════════════════════════════════════════════════════════════════╗
# ║ SETUP: Random number generator & test sizes                                ║
# ╚════════════════════════════════════════════════════════════════════════════╝

const RNG = MersenneTwister(1234)
const TEST_SIZES = [64, 128, 256]

# ╔════════════════════════════════════════════════════════════════════════════╗
# ║ SIZE-SPECIFIC BENCHMARKS                                                   ║
# ╚════════════════════════════════════════════════════════════════════════════╝

for sz in TEST_SIZES
    sz_str = string(sz)

    # Generate test image for this size
    img = randn(RNG, sz, sz)

    # ────────────────────────────────────────────────────────────────────────────
    # Contourlet Transform (CT): forward/inverse
    # ────────────────────────────────────────────────────────────────────────────

    SUITE["CT"][sz_str] = BenchmarkGroup()

    # Setup CT with J=2, L_array=[2, 3]
    p_ct = ContourletParams(J = 2, L_array = [2, 3])
    ws_ct = make_workspace(Float64, (sz, sz), p_ct)
    coeffs_ct_alloc = similar_coefficients(p_ct, (sz, sz))

    # CT Forward: allocating version
    SUITE["CT"][sz_str]["forward"] = @benchmarkable ct_forward($img, $p_ct)

    # CT Forward: in-place with workspace
    SUITE["CT"][sz_str]["forward!"] = @benchmarkable ct_forward!($coeffs_ct_alloc, $img, $ws_ct)

    # Get coefficients for inverse benchmarks
    coeffs_ct = ct_forward(img, p_ct)

    # CT Inverse: allocating version
    SUITE["CT"][sz_str]["inverse"] = @benchmarkable ct_inverse($coeffs_ct)

    # CT Inverse: in-place with workspace
    img_out_ct = similar(img)
    SUITE["CT"][sz_str]["inverse!"] = @benchmarkable ct_inverse!($img_out_ct, $coeffs_ct, $ws_ct)

    # ────────────────────────────────────────────────────────────────────────────
    # Non-Subsampled Contourlet Transform (NSCT): forward/inverse
    # ────────────────────────────────────────────────────────────────────────────

    SUITE["NSCT"][sz_str] = BenchmarkGroup()

    # Setup NSCT with J=2, L_array=[2, 3]
    p_nsct = ContourletParams(J = 2, L_array = [2, 3])
    ws_nsct = make_nsct_workspace(Float64, (sz, sz), p_nsct)
    coeffs_nsct_alloc = similar_nsct_coefficients(p_nsct, (sz, sz))

    # NSCT Forward: allocating version
    SUITE["NSCT"][sz_str]["forward"] = @benchmarkable nsct_forward($img, $p_nsct)

    # NSCT Forward: in-place with workspace
    SUITE["NSCT"][sz_str]["forward!"] = @benchmarkable nsct_forward!($coeffs_nsct_alloc, $img, $ws_nsct)

    # Get NSCT coefficients for inverse benchmark
    coeffs_nsct = nsct_forward(img, p_nsct)

    # NSCT Inverse: allocating version
    SUITE["NSCT"][sz_str]["inverse"] = @benchmarkable nsct_inverse($coeffs_nsct)

    # NSCT Inverse: in-place with workspace
    img_out_nsct = similar(img)
    SUITE["NSCT"][sz_str]["inverse!"] = @benchmarkable nsct_inverse!($img_out_nsct, $coeffs_nsct, $ws_nsct)

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

    # DFB at multiple decomposition levels (1, 2, 3, 4)
    for level in [1, 2, 3, 4]
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
    for level in [1, 2, 3]
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
    k_large = randn(RNG, 7, 7)                           # FFTW backend (>25 taps)
    SUITE["primitives"][sz_str]["conv2d_sep"] =
        @benchmarkable conv2d_sep($img, [1.0, 2.0, 1.0], [1.0, 2.0, 1.0])
    SUITE["primitives"][sz_str]["conv2d_direct"] =
        @benchmarkable conv2d($img, $k_small, (1, 2))
    SUITE["primitives"][sz_str]["conv2d_fftw"] =
        @benchmarkable conv2d($img, $k_large, (4, 4))
end
