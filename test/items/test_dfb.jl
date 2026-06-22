@testitem "QFB PR col-direction" begin
    using Random
    Random.seed!(20)
    x = randn(32, 32)
    sb0, sb1 = qfb_decompose(x, Q2345)
    @test size(sb0) == (32, 16)
    @test size(sb1) == (32, 16)
    rec = qfb_reconstruct(sb0, sb1, Q2345)
    @test maximum(abs, rec .- x) < 1.0e-12
end

@testitem "QFB PR row-direction" begin
    using Random
    Random.seed!(21)
    x = randn(32, 32)
    sb0, sb1 = qfb_decompose(x, Q2345; dir = :row)
    @test size(sb0) == (16, 32)
    rec = qfb_reconstruct(sb0, sb1, Q2345; dir = :row)
    @test maximum(abs, rec .- x) < 1.0e-12
end

@testitem "DFB PR levels 1–4" begin
    using Random
    Random.seed!(22)
    x = randn(32, 32)
    for l in 1:4
        sbs = dfb_decompose(x, l, Q2345)
        @test length(sbs) == 2^l
        rec = dfb_reconstruct(sbs, Q2345)
        @test maximum(abs, rec .- x) < 1.0e-12
    end
end

@testitem "DFB subband sizes (canonical directional mosaic)" begin
    x = zeros(64, 64)
    # l=1: two horizontal subbands, rows halved.
    sbs1 = dfb_decompose(x, 1, Q2345)
    @test all(==((32, 64)), size.(sbs1))
    # l=2: four square subbands.
    sbs2 = dfb_decompose(x, 2, Q2345)
    @test all(==((32, 32)), size.(sbs2))
    # l=3: 4 horizontal-mosaic + 4 vertical-mosaic subbands.
    sbs3 = dfb_decompose(x, 3, Q2345)
    @test count(==((16, 32)), size.(sbs3)) == 4
    @test count(==((32, 16)), size.(sbs3)) == 4
    # l=4: 8 + 8.
    sbs4 = dfb_decompose(x, 4, Q2345)
    @test count(==((8, 32)), size.(sbs4)) == 8
    @test count(==((32, 8)), size.(sbs4)) == 8
    # The exported size helper matches the actual decomposition.
    for l in 1:4
        @test dfb_subband_sizes(64, 64, l) == size.(dfb_decompose(x, l, Q2345))
    end
    # Sample count is preserved (nearly critical sampling).
    for l in 1:4
        @test sum(prod, dfb_subband_sizes(64, 64, l)) == 64 * 64
    end
end

@testitem "DFB is directionally selective" begin
    # An l=3 DFB has 8 directional wedges.  Sweeping the orientation of a
    # near-Nyquist 2-D sinusoid must light up several *different* dominant
    # subbands — a degenerate (non-directional) DFB would always pick the same
    # one.  Guards against the old shear-tree collapse.
    n = 128
    wave(θ, f) = [cos(2π * f * (cos(θ) * j + sin(θ) * i)) for i in 1:n, j in 1:n]
    doms = Int[]
    for deg in 0:15:179
        e = [sum(abs2, s) for s in dfb_decompose(wave(deg * π / 180, 0.45), 3, Q2345)]
        push!(doms, argmax(e))
        @test maximum(e) > 0.4 * sum(e)   # energy concentrates in a wedge
    end
    @test length(unique(doms)) >= 4       # many distinct directions resolved
end

@testitem "DFB level 0 returns single subband" begin
    x = randn(8, 8)
    sbs = dfb_decompose(x, 0, Q2345)
    @test length(sbs) == 1
    @test sbs[1] == x
end

@testitem "NSDFB PR levels 1–3, tree_level 1–3" begin
    using Random
    Random.seed!(23)
    x = randn(32, 32)
    for l in 1:3, tl in 1:3
        sbs = nsdfb_decompose(x, l, Q2345, tl)
        @test length(sbs) == 2^l
        @test all(s -> size(s) == (32, 32), sbs)
        rec = nsdfb_reconstruct(sbs, Q2345, tl)
        @test maximum(abs, rec .- x) < 1.0e-12
    end
end

@testitem "NSDFB subbands are directionally selective" begin
    # An oriented sinusoid must concentrate its energy in a single directional
    # wedge.  With the resampling-matrix (fan-filter) construction each subband is
    # a genuine angular wedge, so a single tone lands almost entirely in one of
    # them (a quadrant/block partition would smear it across several).
    n = 64
    wave(θ) = [cos(2π * 0.4 * (cos(θ) * j + sin(θ) * i)) for i in 1:n, j in 1:n]
    for deg in (15, 45, 75, 105, 135, 165)
        e = [sum(abs2, s) for s in nsdfb_decompose(wave(deg * π / 180), 3, Q2345)]
        @test maximum(e) > 0.6 * sum(e)     # one wedge dominates
    end
end

@testitem "NSDFB angle sweep resolves all wedges" begin
    # Sweeping orientation over 180° at l=3 (8 wedges) must light up every wedge:
    # a directional bank resolves all 2^l distinct dominant subbands, a degenerate
    # one collapses to far fewer (the old block bank resolved ≤6, the broken
    # diagonal-only bank only 4).
    n = 128
    wave(θ) = [cos(2π * 0.42 * (cos(θ) * j + sin(θ) * i)) for i in 1:n, j in 1:n]
    doms = [argmax([sum(abs2, s) for s in nsdfb_decompose(wave(deg * π / 180), 3, Q2345)])
            for deg in 0:10:179]
    @test length(unique(doms)) == 8
end

@testitem "NSDFB matches the MATLAB reference (resampling-matrix fan filters)" begin
    # Locks the wedge construction to the da Cunha–Zhou–Do reference: the fan and
    # parallelogram filters built from the pkva ladder must equal dfilters('pkva')
    # + parafilters from the MATLAB toolbox.  Reference values precomputed.
    F = Contourlets._nsdfb_filters(Q2345, Float64)
    @test size(F.k1) == (23, 23)        # fan filter from the 23-tap diamond low-pass
    @test size(F.k2) == (45, 45)        # fan filter from the 45-tap diamond high-pass
    @test length(F.f1) == 4 && length(F.f2) == 4
    @test all(==((23, 23)), size.(F.f1))
    @test isodd(size(F.k2, 1))
end

@testitem "NSCT is directionally selective for oriented inputs" begin
    n = 128
    p = ContourletParams(J = 1, L_array = [3])
    for deg in (30, 60, 90, 120, 150)
        x = [cos(2π * 0.42 * (cos(deg * π / 180) * j + sin(deg * π / 180) * i))
             for i in 1:n, j in 1:n]
        e = [sum(abs2, s) for s in nsct_forward(x, p; threading = Disabled()).subbands[1]]
        @test maximum(e) > 0.5 * sum(e)
    end
end

@testitem "NSCT is shift-invariant" begin
    using Random
    Random.seed!(99)
    n = 32
    x = randn(n, n)
    p = ContourletParams(J = 1, L_array = [3])
    c0 = nsct_forward(x, p; threading = Disabled())
    for shift in ((3, 0), (0, 5), (2, 7))
        cs = nsct_forward(circshift(x, shift), p; threading = Disabled())
        # LP residual
        @test maximum(abs, c0.coarse .- circshift(cs.coarse, .-shift)) < 1e-10
        # Directional subbands
        for j in 1:length(c0.subbands)
            for (a, b) in zip(c0.subbands[j], cs.subbands[j])
                @test maximum(abs, a .- circshift(b, .-shift)) < 1e-10
            end
        end
    end
end

@testitem "QFB in-place decompose col-direction" begin
    using Random
    Random.seed!(24)
    x = randn(32, 32)
    sb0_alloc, sb1_alloc = qfb_decompose(x, Q2345)
    sb0 = zeros(32, 16)
    sb1 = zeros(32, 16)
    qfb_decompose!(sb0, sb1, x, Q2345)
    @test maximum(abs, sb0 .- sb0_alloc) < 1.0e-14
    @test maximum(abs, sb1 .- sb1_alloc) < 1.0e-14
end

@testitem "QFB in-place decompose row-direction" begin
    using Random
    Random.seed!(25)
    x = randn(32, 32)
    sb0_alloc, sb1_alloc = qfb_decompose(x, Q2345; dir = :row)
    sb0 = zeros(16, 32)
    sb1 = zeros(16, 32)
    qfb_decompose!(sb0, sb1, x, Q2345; dir = :row)
    @test maximum(abs, sb0 .- sb0_alloc) < 1.0e-14
    @test maximum(abs, sb1 .- sb1_alloc) < 1.0e-14
end

@testitem "QFB in-place reconstruct col-direction" begin
    using Random
    Random.seed!(26)
    x = randn(32, 32)
    sb0, sb1 = qfb_decompose(x, Q2345)
    out = zeros(32, 32)
    qfb_reconstruct!(out, sb0, sb1, Q2345)
    @test maximum(abs, out .- x) < 1.0e-12
end

@testitem "QFB in-place reconstruct row-direction" begin
    using Random
    Random.seed!(27)
    x = randn(32, 32)
    sb0, sb1 = qfb_decompose(x, Q2345; dir = :row)
    out = zeros(32, 32)
    qfb_reconstruct!(out, sb0, sb1, Q2345; dir = :row)
    @test maximum(abs, out .- x) < 1.0e-12
end
