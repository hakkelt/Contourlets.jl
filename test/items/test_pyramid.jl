@testitem "LP PR — Float64" tags = [:pyramid] begin
    using Random
    Random.seed!(10)
    x = randn(32, 32)
    c, bp = Contourlets.lp_decompose(x, CDF97)
    @test size(c) == (16, 16)
    @test size(bp) == (32, 32)
    rec = Contourlets.lp_reconstruct(c, bp, CDF97)
    @test maximum(abs, rec .- x) < 1.0e-12
end

@testitem "LP PR — Float32" tags = [:pyramid] begin
    using Random
    Random.seed!(11)
    x = randn(Float32, 32, 32)
    c, bp = Contourlets.lp_decompose(x, CDF97)
    @test eltype(c) == Float32
    rec = Contourlets.lp_reconstruct(c, bp, CDF97)
    @test maximum(abs, rec .- x) < 1.0e-4
end

@testitem "LP in-place round-trip" tags = [:pyramid] begin
    using Random
    Random.seed!(12)
    x = randn(32, 32)
    c = zeros(16, 16); bp = zeros(32, 32)
    Contourlets.lp_decompose!(c, bp, x, CDF97)
    rec = similar(x)
    Contourlets.lp_reconstruct!(rec, c, bp, CDF97)
    @test maximum(abs, rec .- x) < 1.0e-12
end

@testitem "NSP PR — multiple levels" tags = [:pyramid] begin
    using Random
    Random.seed!(13)
    x = randn(32, 32)
    for lv in 1:3
        c, bp = Contourlets.nsp_decompose(x, CDF97, lv)
        @test size(c) == size(x)
        @test size(bp) == size(x)
        rec = Contourlets.nsp_reconstruct(c, bp, CDF97, lv)
        @test maximum(abs, rec .- x) < 1.0e-12
    end
end

@testitem "NSP in-place decompose/reconstruct" tags = [:pyramid] begin
    using Random
    Random.seed!(14)
    x = randn(32, 32)
    c = similar(x)
    bp = similar(x)
    tmp = similar(x)
    Contourlets.nsp_decompose!(c, bp, x, CDF97, 1; tmp = tmp)
    rec = similar(x)
    Contourlets.nsp_reconstruct!(rec, c, bp, CDF97, 1; tmp = tmp)
    @test maximum(abs, rec .- x) < 1.0e-12
end

@testitem "NSP in-place level 2 and 3" tags = [:pyramid] begin
    using Random
    Random.seed!(15)
    x = randn(32, 32)
    for lv in 2:3
        c = similar(x); bp = similar(x); tmp = similar(x)
        Contourlets.nsp_decompose!(c, bp, x, CDF97, lv)
        rec = similar(x)
        Contourlets.nsp_reconstruct!(rec, c, bp, CDF97, lv)
        @test maximum(abs, rec .- x) < 1.0e-12
    end
end

@testitem "NSP decompose invalid level" tags = [:pyramid] begin
    x = randn(16, 16)
    @test_throws ArgumentError Contourlets.nsp_decompose(x, CDF97, 0)
end

@testitem "LP round trip on odd sizes" tags = [:pyramid] begin
    using Random
    x = randn(Xoshiro(1), 31, 33)
    c, bp = Contourlets.lp_decompose(x, CDF97)
    @test size(c) == (16, 17) && size(bp) == (31, 33)
    @test Contourlets.lp_reconstruct(c, bp, CDF97) ≈ x atol = 1.0e-12
    c2, bp2 = zeros(16, 17), zeros(31, 33)
    Contourlets.lp_decompose!(c2, bp2, x, CDF97)
    @test c2 ≈ c && bp2 ≈ bp
    y = zeros(31, 33)
    @test Contourlets.lp_reconstruct!(y, c2, bp2, CDF97) ≈ x atol = 1.0e-12
end

@testitem "CT on odd level sizes, and its size rule" tags = [:pyramid] begin
    using Random
    x = randn(Xoshiro(2), 100, 100)       # 100 → 50 → 25
    t = CT(scales = 3, direction_levels = [0, 2, 3])
    @test synthesize(t, analyze(t, x)) ≈ x atol = 1.0e-12
    y = randn(Xoshiro(3), 31, 33)
    t2 = CT(scales = 2, direction_levels = [0, 0])
    @test synthesize(t2, analyze(t2, y)) ≈ y atol = 1.0e-12
    @test_throws "at scale 1 of 3 (1 the coarsest) axis 1 has bandpass length 25, which its 1-level directional filter bank needs divisible by 2; the nearest image length that fits is 96" analyze(CT(scales = 3), x)
    @test_throws "does not fit" make_workspace(CT(scales = 3), Float64, (100, 100))
end
