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
    # A horizontally-varying pattern (vertical edges) and a vertically-varying
    # pattern (horizontal edges) must concentrate their energy in different
    # directional subbands.  This guards against a degenerate DFB whose stages
    # collapse to a single direction (regression test).
    n = 32
    horiz = [sinpi(8 * j / n) for _ in 1:n, j in 1:n]   # varies along columns
    vert = [sinpi(8 * i / n) for i in 1:n, _ in 1:n]    # varies along rows
    eh = [sum(abs2, s) for s in nsdfb_decompose(horiz, 2, Q2345, 1)]
    ev = [sum(abs2, s) for s in nsdfb_decompose(vert, 2, Q2345, 1)]
    ph = eh ./ sum(eh)
    pv = ev ./ sum(ev)
    # The two orientations produce clearly different energy distributions across
    # the directional subbands (a degenerate single-direction DFB would give
    # nearly identical distributions).
    @test sum(abs, ph .- pv) > 0.2
    # Energy is concentrated in a few subbands, not spread uniformly.
    @test maximum(ph) > 0.5
    @test maximum(pv) > 0.5
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
