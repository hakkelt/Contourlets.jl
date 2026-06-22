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
    # A diagonal-varying pattern and an anti-diagonal-varying pattern must
    # concentrate their energy in different directional subbands.  With the fan
    # partition (depth-1 direction (1,1), depth-2 direction (1,-1)), a diagonal
    # signal (ω_i+ω_j ≈ high, ω_i-ω_j = 0) and an anti-diagonal signal
    # (ω_i+ω_j = 0, ω_i-ω_j ≈ high) land in orthogonal branches of the tree.
    n = 32
    # cospi(3/8*(i+j)) → frequency at ω_i+ω_j = 3π/4 (HP at depth 1), ω_i-ω_j = 0 (LP at depth 2)
    diag = [cospi(3 / 8 * (i + j)) for i in 1:n, j in 1:n]
    # cospi(3/8*(i-j)) → frequency at ω_i+ω_j = 0 (LP at depth 1), ω_i-ω_j = 3π/4 (HP at depth 2)
    anti = [cospi(3 / 8 * (i - j)) for i in 1:n, j in 1:n]
    ed = [sum(abs2, s) for s in nsdfb_decompose(diag, 2, Q2345, 1)]
    ea = [sum(abs2, s) for s in nsdfb_decompose(anti, 2, Q2345, 1)]
    pd = ed ./ sum(ed)
    pa = ea ./ sum(ea)
    # The two orientations produce clearly different energy distributions across
    # the directional subbands (a degenerate single-direction DFB would give
    # nearly identical distributions).
    @test sum(abs, pd .- pa) > 0.2
    # Energy is concentrated in a few subbands, not spread uniformly.
    @test maximum(pd) > 0.5
    @test maximum(pa) > 0.5
end

@testitem "NSDFB diagonal sinusoid concentrates in a single wedge" begin
    # A quadrant partition (broken) spreads diagonal energy across multiple
    # subbands; a fan/wedge partition must concentrate it in one.
    n = 64
    wave(θ) = [cos(2π * 0.4 * (cos(θ) * j + sin(θ) * i)) for i in 1:n, j in 1:n]
    for deg in (45, 135)
        e = [sum(abs2, s) for s in nsdfb_decompose(wave(deg * π / 180), 2, Q2345, 1)]
        @test maximum(e) > 0.4 * sum(e)   # one wedge dominates
    end
end

@testitem "NSDFB angle sweep covers all 4 wedges" begin
    # Sweeping orientation 0–179° in 15° steps for l=2 must visit all 4 distinct
    # dominant subband indices.  A quadrant/block partition resolves at most 2
    # distinct dominant indices for diagonal sinusoids.
    n = 128
    wave(θ) = [cos(2π * 0.42 * (cos(θ) * j + sin(θ) * i)) for i in 1:n, j in 1:n]
    doms = [argmax([sum(abs2, s) for s in nsdfb_decompose(wave(deg * π / 180), 2, Q2345, 1)])
            for deg in 0:15:179]
    @test length(unique(doms)) >= 3   # at least 3 out of 4 wedges resolved in the sweep
end

@testitem "NSCT is directionally selective for oriented inputs" begin
    # For each orientation, the NSCT (J=1, L=2) must have a clear dominant
    # directional subband — just like the decimated CT.  We don't require the
    # same subband *index* (CT and NSCT have different subband orderings), but
    # both must concentrate > 40% of their DFB energy in a single channel.
    n = 128
    p = ContourletParams(J = 1, L_array = [2])
    for deg in (30, 60, 90, 120, 150)
        x = [cos(2π * 0.42 * (cos(deg * π / 180) * j + sin(deg * π / 180) * i))
             for i in 1:n, j in 1:n]
        ct_e = [sum(abs2, s) for s in ct_forward(x, p; threading = Disabled()).subbands[1]]
        ns_e = [sum(abs2, s) for s in nsct_forward(x, p; threading = Disabled()).subbands[1]]
        @test maximum(ct_e) > 0.3 * sum(ct_e)
        @test maximum(ns_e) > 0.3 * sum(ns_e)
    end
end

@testitem "NSDFB frequency support is a wedge (not a block)" begin
    # The impulse response of each NSDFB channel, when DFT'd, must have its
    # spectral mass concentrated in a wedge through the origin (a bowtie), not
    # in an axis-aligned quadrant block.  We use the "off-axis centroid" metric:
    # the centroid of |H_k(ω)|^2 must be away from both frequency axes.
    using FFTW
    n = 64
    impulse = zeros(n, n); impulse[1, 1] = 1.0
    sbs = nsdfb_decompose(impulse, 2, Q2345, 1)
    for sb in sbs
        S = abs2.(fftshift(fft(sb)))
        total = sum(S)
        # Frequency-axis grid: ω ∈ [-π, π), sampled at 2π/n steps
        ωi = [(i - n ÷ 2 - 1) * 2π / n for i in 1:n]
        ωj = [(j - n ÷ 2 - 1) * 2π / n for j in 1:n]
        # Centroid of ω_i + ω_j and ω_i - ω_j
        centroid_sum = sum(S[i, j] * abs(ωi[i] + ωj[j]) for i in 1:n, j in 1:n) / total
        centroid_diff = sum(S[i, j] * abs(ωi[i] - ωj[j]) for i in 1:n, j in 1:n) / total
        # A wedge (bowtie through origin) has its centroid off the frequency axes.
        # At least one diagonal direction must carry substantial mass.
        @test max(centroid_sum, centroid_diff) > 0.3
    end
end

@testitem "NSCT is shift-invariant" begin
    using Random
    Random.seed!(99)
    n = 32
    x = randn(n, n)
    p = ContourletParams(J = 1, L_array = [2])
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
