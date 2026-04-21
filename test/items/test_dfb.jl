@testitem "QFB PR col-direction" begin
    using Contourlets, Random
    Random.seed!(20)
    x = randn(32, 32)
    sb0, sb1 = qfb_decompose(x, Q2345)
    @test size(sb0) == (32, 16)
    @test size(sb1) == (32, 16)
    rec = qfb_reconstruct(sb0, sb1, Q2345)
    @test maximum(abs, rec .- x) < 1.0e-12
end

@testitem "QFB PR row-direction" begin
    using Contourlets, Random
    Random.seed!(21)
    x = randn(32, 32)
    sb0, sb1 = qfb_decompose(x, Q2345; dir = :row)
    @test size(sb0) == (16, 32)
    rec = qfb_reconstruct(sb0, sb1, Q2345; dir = :row)
    @test maximum(abs, rec .- x) < 1.0e-12
end

@testitem "DFB PR levels 1–4" begin
    using Contourlets, Random
    Random.seed!(22)
    x = randn(32, 32)
    for l in 1:4
        sbs = dfb_decompose(x, l, Q2345)
        @test length(sbs) == 2^l
        rec = dfb_reconstruct(sbs, Q2345)
        @test maximum(abs, rec .- x) < 1.0e-12
    end
end

@testitem "DFB subband sizes follow parabolic scaling" begin
    using Contourlets
    x = zeros(32, 32)
    sbs1 = dfb_decompose(x, 1, Q2345);  @test size(sbs1[1]) == (32, 16)
    sbs2 = dfb_decompose(x, 2, Q2345);  @test size(sbs2[1]) == (16, 16)
    sbs3 = dfb_decompose(x, 3, Q2345);  @test size(sbs3[1]) == (16, 8)
    sbs4 = dfb_decompose(x, 4, Q2345);  @test size(sbs4[1]) == (8, 8)
end

@testitem "DFB level 0 returns single subband" begin
    using Contourlets
    x = randn(8, 8)
    sbs = dfb_decompose(x, 0, Q2345)
    @test length(sbs) == 1
    @test sbs[1] == x
end

@testitem "NSDFB PR levels 1–3, tree_level 1–3" begin
    using Contourlets, Random
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
