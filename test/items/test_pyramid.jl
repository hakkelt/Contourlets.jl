@testitem "LP PR — Float64" begin
    using Random
    Random.seed!(10)
    x = randn(32, 32)
    c, bp = lp_decompose(x, CDF97)
    @test size(c) == (16, 16)
    @test size(bp) == (32, 32)
    rec = lp_reconstruct(c, bp, CDF97)
    @test maximum(abs, rec .- x) < 1.0e-12
end

@testitem "LP PR — Float32" begin
    using Random
    Random.seed!(11)
    x = randn(Float32, 32, 32)
    c, bp = lp_decompose(x, CDF97)
    @test eltype(c) == Float32
    rec = lp_reconstruct(c, bp, CDF97)
    @test maximum(abs, rec .- x) < 1.0e-4
end

@testitem "LP in-place round-trip" begin
    using Random
    Random.seed!(12)
    x = randn(32, 32)
    c = zeros(16, 16); bp = zeros(32, 32)
    lp_decompose!(c, bp, x, CDF97)
    rec = similar(x)
    lp_reconstruct!(rec, c, bp, CDF97)
    @test maximum(abs, rec .- x) < 1.0e-12
end

@testitem "NSP PR — multiple levels" begin
    using Random
    Random.seed!(13)
    x = randn(32, 32)
    for lv in 1:3
        c, bp = nsp_decompose(x, CDF97, lv)
        @test size(c) == size(x)
        @test size(bp) == size(x)
        rec = nsp_reconstruct(c, bp, CDF97, lv)
        @test maximum(abs, rec .- x) < 1.0e-12
    end
end

@testitem "NSP in-place decompose/reconstruct" begin
    using Random
    Random.seed!(14)
    x = randn(32, 32)
    c = similar(x)
    bp = similar(x)
    tmp = similar(x)
    nsp_decompose!(c, bp, x, CDF97, 1; tmp = tmp)
    rec = similar(x)
    nsp_reconstruct!(rec, c, bp, CDF97, 1; tmp = tmp)
    @test maximum(abs, rec .- x) < 1.0e-12
end

@testitem "NSP in-place level 2 and 3" begin
    using Random
    Random.seed!(15)
    x = randn(32, 32)
    for lv in 2:3
        c = similar(x); bp = similar(x); tmp = similar(x)
        nsp_decompose!(c, bp, x, CDF97, lv)
        rec = similar(x)
        nsp_reconstruct!(rec, c, bp, CDF97, lv)
        @test maximum(abs, rec .- x) < 1.0e-12
    end
end

@testitem "NSP decompose invalid level" begin
    x = randn(16, 16)
    @test_throws ArgumentError nsp_decompose(x, CDF97, 0)
end
