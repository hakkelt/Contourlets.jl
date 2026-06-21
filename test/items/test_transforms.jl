@testitem "CT forward/inverse PR" begin
    using Random
    Random.seed!(30)
    x = randn(32, 32)
    params = ContourletParams(J = 2, L_array = [2, 3])
    c = ct_forward(x, params)
    @test c isa ContourletCoefficients
    rec = ct_inverse(c)
    @test maximum(abs, rec .- x) < 1.0e-12
end

@testitem "CT coeff structure" begin
    x = zeros(32, 32); x[16, 16] = 1.0
    p = ContourletParams(J = 2, L_array = [2, 3])
    c = ct_forward(x, p)
    @test length(c.subbands) == 2
    @test length(c.subbands[1]) == 4   # 2^2 directional subbands at level 1
    @test length(c.subbands[2]) == 8   # 2^3 directional subbands at level 2
    @test size(c.coarse) == (8, 8)
end

@testitem "CT Float32 round-trip" begin
    using Random
    Random.seed!(31)
    x = randn(Float32, 32, 32)
    p = ContourletParams(J = 1, L_array = [2])
    c = ct_forward(x, p)
    rec = ct_inverse(c)
    @test eltype(rec) == Float32
    @test maximum(abs, rec .- x) < 1.0e-4
end

@testitem "NSCT forward/inverse PR" begin
    using Random
    Random.seed!(32)
    x = randn(32, 32)
    params = ContourletParams(J = 2, L_array = [2, 3])
    nc = nsct_forward(x, params)
    @test nc isa NSCTCoefficients
    rec = nsct_inverse(nc)
    @test maximum(abs, rec .- x) < 1.0e-12
end

@testitem "NSCT shift invariance" begin
    using Random
    Random.seed!(38)
    x = randn(32, 32)
    p = ContourletParams(J = 2, L_array = [2, 3])
    c0 = nsct_forward(x, p)
    for s in ((1, 0), (0, 1), (3, 5), (7, 11))
        cs = nsct_forward(circshift(x, s), p)
        # Coefficients of the shifted image equal the shifted coefficients,
        # with no redistribution of energy across subbands (spec §5 Step 4).
        @test maximum(abs, cs.coarse .- circshift(c0.coarse, s)) < 1.0e-12
        for j in 1:2, k in eachindex(c0.subbands[j])
            @test maximum(abs, cs.subbands[j][k] .- circshift(c0.subbands[j][k], s)) < 1.0e-12
        end
    end
end

@testitem "NSCT subband energy is shift invariant" begin
    using Random
    Random.seed!(39)
    x = randn(32, 32)
    p = ContourletParams(J = 2, L_array = [2, 3])
    e0 = [sum(abs2, sb) for sbs in nsct_forward(x, p).subbands for sb in sbs]
    es = [sum(abs2, sb) for sbs in nsct_forward(circshift(x, (4, 9)), p).subbands for sb in sbs]
    @test maximum(abs, e0 .- es) < 1.0e-8
end

@testitem "NSCT all subbands same size as input" begin
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
    using Random
    Random.seed!(33)
    x = randn(32, 32)
    p = ContourletParams(J = 0, L_array = Int[])
    c = ct_forward(x, p)
    @test c.coarse == x
    rec = ct_inverse(c)
    @test maximum(abs, rec .- x) < 1.0e-14
end

@testitem "parabolic_levels helper" begin
    ls = parabolic_levels(4)
    @test length(ls) == 4
    @test all(x -> x >= 1, ls)
    @test issorted(ls; rev = true)  # coarser levels have more directions
end

@testitem "CT in-place forward/inverse with workspace" begin
    using Random
    Random.seed!(34)
    x = randn(32, 32)
    p = ContourletParams(J = 2, L_array = [2, 3])
    ws = make_workspace(p, (32, 32))
    coeffs = similar_coefficients(p, (32, 32))
    ct_forward!(coeffs, x, ws)
    rec = similar(x)
    ct_inverse!(rec, coeffs, ws)
    @test maximum(abs, rec .- x) < 1.0e-12
end

@testitem "CT in-place forward matches allocating" begin
    using Random
    Random.seed!(35)
    x = randn(32, 32)
    p = ContourletParams(J = 2, L_array = [2, 3])
    ws = make_workspace(p, (32, 32))
    coeffs = similar_coefficients(p, (32, 32))
    ct_forward!(coeffs, x, ws)
    coeffs_alloc = ct_forward(x, p)
    @test maximum(abs, coeffs.coarse .- coeffs_alloc.coarse) < 1.0e-12
    for j in 1:2, k in 1:length(coeffs.subbands[j])
        @test maximum(abs, coeffs.subbands[j][k] .- coeffs_alloc.subbands[j][k]) < 1.0e-12
    end
end

@testitem "NSCT in-place forward/inverse with workspace" begin
    using Random
    Random.seed!(36)
    x = randn(32, 32)
    p = ContourletParams(J = 2, L_array = [2, 3])
    ws = make_nsct_workspace(p, (32, 32))
    coeffs = similar_nsct_coefficients(p, (32, 32))
    nsct_forward!(coeffs, x, ws)
    rec = similar(x)
    nsct_inverse!(rec, coeffs, ws)
    @test maximum(abs, rec .- x) < 1.0e-12
end

@testitem "NSCT in-place forward matches allocating" begin
    using Random
    Random.seed!(37)
    x = randn(32, 32)
    p = ContourletParams(J = 2, L_array = [2, 3])
    ws = make_nsct_workspace(p, (32, 32))
    coeffs = similar_nsct_coefficients(p, (32, 32))
    nsct_forward!(coeffs, x, ws)
    coeffs_alloc = nsct_forward(x, p)
    @test maximum(abs, coeffs.coarse .- coeffs_alloc.coarse) < 1.0e-12
    for j in 1:2, k in 1:length(coeffs.subbands[j])
        @test isapprox(coeffs.subbands[j][k], coeffs_alloc.subbands[j][k]; atol = 1.0e-10)
    end
end

@testitem "similar_nsct_coefficients sizes" begin
    p = ContourletParams(J = 2, L_array = [2, 3])
    nc = similar_nsct_coefficients(p, (64, 64))
    @test size(nc.coarse) == (64, 64)
    @test length(nc.subbands[1]) == 4   # 2^2
    @test length(nc.subbands[2]) == 8   # 2^3
    for sbs in nc.subbands
        for sb in sbs
            @test size(sb) == (64, 64)
        end
    end
end

@testitem "similar_coefficients sizes" begin
    p = ContourletParams(J = 2, L_array = [2, 3])
    c = similar_coefficients(p, (32, 32))
    @test length(c.subbands[1]) == 4
    @test length(c.subbands[2]) == 8
    @test size(c.coarse) == (8, 8)
end
