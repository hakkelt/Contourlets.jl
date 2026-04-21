@testitem "CT forward/inverse PR" begin
    using Contourlets, Random
    Random.seed!(30)
    x = randn(32, 32)
    params = ContourletParams(J = 2, L_array = [2, 3])
    c = ct_forward(x, params)
    @test c isa ContourletCoefficients
    rec = ct_inverse(c)
    @test maximum(abs, rec .- x) < 1.0e-12
end

@testitem "CT coeff structure" begin
    using Contourlets
    x = zeros(32, 32); x[16, 16] = 1.0
    p = ContourletParams(J = 2, L_array = [2, 3])
    c = ct_forward(x, p)
    @test length(c.subbands) == 2
    @test length(c.subbands[1]) == 4   # 2^2 directional subbands at level 1
    @test length(c.subbands[2]) == 8   # 2^3 directional subbands at level 2
    @test size(c.coarse) == (8, 8)
end

@testitem "CT Float32 round-trip" begin
    using Contourlets, Random
    Random.seed!(31)
    x = randn(Float32, 32, 32)
    p = ContourletParams(J = 1, L_array = [2])
    c = ct_forward(x, p)
    rec = ct_inverse(c)
    @test eltype(rec) == Float32
    @test maximum(abs, rec .- x) < 1.0e-4
end

@testitem "NSCT forward/inverse PR" begin
    using Contourlets, Random
    Random.seed!(32)
    x = randn(32, 32)
    params = ContourletParams(J = 2, L_array = [2, 3])
    nc = nsct_forward(x, params)
    @test nc isa NSCTCoefficients
    rec = nsct_inverse(nc)
    @test maximum(abs, rec .- x) < 1.0e-12
end

@testitem "NSCT all subbands same size as input" begin
    using Contourlets
    x = zeros(32, 32)
    p = ContourletParams(J = 2, L_array = [2, 3])
    nc = nsct_forward(x, p)
    for sbs in nc.subbands
        for sb in sbs
            @test size(sb) == size(x)
        end
    end
end

@testitem "CT J=0 trivial reconstruction" begin
    using Contourlets, Random
    Random.seed!(33)
    x = randn(32, 32)
    p = ContourletParams(J = 0, L_array = Int[])
    c = ct_forward(x, p)
    @test c.coarse == x
    rec = ct_inverse(c)
    @test maximum(abs, rec .- x) < 1.0e-14
end

@testitem "parabolic_levels helper" begin
    using Contourlets
    ls = parabolic_levels(4)
    @test length(ls) == 4
    @test all(x -> x >= 1, ls)
    @test issorted(ls; rev = true)  # coarser levels have more directions
end
