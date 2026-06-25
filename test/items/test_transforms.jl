@testitem "CT forward/inverse PR" tags = [:ct] begin
    using Random
    Random.seed!(30)
    x = randn(32, 32)
    params = ContourletParams(J = 2, L_array = [2, 3])
    c = ct_forward(x, params)
    @test c isa ContourletCoefficients
    rec = ct_inverse(c, params)
    @test maximum(abs, rec .- x) < 1.0e-12
end

@testitem "CT coeff structure" tags = [:ct] begin
    x = zeros(32, 32); x[16, 16] = 1.0
    p = ContourletParams(J = 2, L_array = [2, 3])
    c = ct_forward(x, p)
    @test length(c.subbands) == 2
    @test length(c.subbands[1]) == 4   # 2^2 directional subbands at level 1
    @test length(c.subbands[2]) == 8   # 2^3 directional subbands at level 2
    @test size(c.coarse) == (8, 8)
end

@testitem "CT Float32 round-trip" tags = [:ct] begin
    using Random
    Random.seed!(31)
    x = randn(Float32, 32, 32)
    p = ContourletParams(J = 1, L_array = [2])
    c = ct_forward(x, p)
    rec = ct_inverse(c, p)
    @test eltype(rec) == Float32
    @test maximum(abs, rec .- x) < 1.0e-4
end

@testitem "NSCT forward/inverse PR" tags = [:nsct] begin
    using Random
    Random.seed!(32)
    x = randn(32, 32)
    params = ContourletParams(J = 2, L_array = [2, 3])
    nc = nsct_forward(x, params)
    @test nc isa NSCTCoefficients
    rec = nsct_inverse(nc, params)
    @test maximum(abs, rec .- x) < 1.0e-12
end

@testitem "NSCT shift invariance" tags = [:nsct] begin
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

@testitem "NSCT subband energy is shift invariant" tags = [:nsct] begin
    using Random
    Random.seed!(39)
    x = randn(32, 32)
    p = ContourletParams(J = 2, L_array = [2, 3])
    e0 = [sum(abs2, sb) for sbs in nsct_forward(x, p).subbands for sb in sbs]
    es = [sum(abs2, sb) for sbs in nsct_forward(circshift(x, (4, 9)), p).subbands for sb in sbs]
    @test maximum(abs, e0 .- es) < 1.0e-8
end

@testitem "NSCT all subbands same size as input" tags = [:nsct] begin
    x = zeros(32, 32)
    p = ContourletParams(J = 2, L_array = [2, 3])
    nc = nsct_forward(x, p)
    for sbs in nc.subbands
        for sb in sbs
            @test size(sb) == size(x)
        end
    end
end

@testitem "ContourletParams J=0 throws" tags = [:ct] begin
    @test_throws ArgumentError ContourletParams(J = 0, L_array = Int[])
end

@testitem "parabolic_levels helper" tags = [:ct] begin
    ls = parabolic_levels(4)
    @test length(ls) == 4
    @test all(x -> x >= 1, ls)
    @test issorted(ls; rev = true)  # coarser levels have more directions
end

@testitem "CT in-place forward/inverse with workspace" tags = [:ct] begin
    using Random
    Random.seed!(34)
    x = randn(32, 32)
    p = ContourletParams(J = 2, L_array = [2, 3])
    ws = make_workspace(p, (32, 32))
    coeffs = similar_coefficients(p, (32, 32))
    ct_forward!(coeffs, x, p; workspace = ws)
    rec = similar(x)
    ct_inverse!(rec, coeffs, p; workspace = ws)
    @test maximum(abs, rec .- x) < 1.0e-12
end

@testitem "CT in-place forward matches allocating" tags = [:ct] begin
    using Random
    Random.seed!(35)
    x = randn(32, 32)
    p = ContourletParams(J = 2, L_array = [2, 3])
    ws = make_workspace(p, (32, 32))
    coeffs = similar_coefficients(p, (32, 32))
    ct_forward!(coeffs, x, p; workspace = ws)
    coeffs_alloc = ct_forward(x, p)
    @test maximum(abs, coeffs.coarse .- coeffs_alloc.coarse) < 1.0e-12
    for j in 1:2, k in 1:length(coeffs.subbands[j])
        @test maximum(abs, coeffs.subbands[j][k] .- coeffs_alloc.subbands[j][k]) < 1.0e-12
    end
end

@testitem "NSCT in-place forward/inverse with workspace" tags = [:nsct] begin
    using Random
    Random.seed!(36)
    x = randn(32, 32)
    p = ContourletParams(J = 2, L_array = [2, 3])
    ws = make_nsct_workspace(p, (32, 32))
    coeffs = similar_nsct_coefficients(p, (32, 32))
    nsct_forward!(coeffs, x, p; workspace = ws)
    rec = similar(x)
    nsct_inverse!(rec, coeffs, p; workspace = ws)
    @test maximum(abs, rec .- x) < 1.0e-12
end

@testitem "NSCT in-place forward matches allocating" tags = [:nsct] begin
    using Random
    Random.seed!(37)
    x = randn(32, 32)
    p = ContourletParams(J = 2, L_array = [2, 3])
    ws = make_nsct_workspace(p, (32, 32))
    coeffs = similar_nsct_coefficients(p, (32, 32))
    nsct_forward!(coeffs, x, p; workspace = ws)
    coeffs_alloc = nsct_forward(x, p)
    @test maximum(abs, coeffs.coarse .- coeffs_alloc.coarse) < 1.0e-12
    for j in 1:2, k in 1:length(coeffs.subbands[j])
        @test isapprox(coeffs.subbands[j][k], coeffs_alloc.subbands[j][k]; atol = 1.0e-10)
    end
end

@testitem "similar_nsct_coefficients sizes" tags = [:nsct] begin
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

@testitem "similar_coefficients sizes" tags = [:ct] begin
    p = ContourletParams(J = 2, L_array = [2, 3])
    c = similar_coefficients(p, (32, 32))
    @test length(c.subbands[1]) == 4
    @test length(c.subbands[2]) == 8
    @test size(c.coarse) == (8, 8)
end

@testitem "parabolic_levels J=0 returns empty" tags = [:ct] begin
    ls = parabolic_levels(0)
    @test ls == Int[]
    ls1 = parabolic_levels(0, 2)
    @test ls1 == Int[]
end

@testitem "parabolic_levels large l_j0" tags = [:ct] begin
    ls = parabolic_levels(5, 2)
    @test length(ls) == 5
    @test ls[1] >= ls[end]   # finer scale ≥ coarser scale in direction count
end

@testitem "Enabled threading policy" tags = [:ct] begin
    @test Contourlets._use_threading(Enabled(), Float64) === true
    @test Contourlets._use_threading(Enabled(), ComplexF64) === true
end

@testitem "Disabled threading policy" tags = [:ct] begin
    @test Contourlets._use_threading(Disabled(), Float64) === false
    @test Contourlets._use_threading(Disabled(), ComplexF64) === false
end

@testitem "Auto threading policy dispatch" tags = [:ct] begin
    @test Contourlets._use_threading(Auto(), Float64) === false
    @test Contourlets._use_threading(Auto(), ComplexF64) === true
    @test Contourlets._use_threading(Auto(), Float32) === false
    @test Contourlets._use_threading(Auto(), ComplexF32) === true
end

@testitem "CT with threading=Enabled() and Real data" tags = [:ct] begin
    using Random
    Random.seed!(90)
    x = randn(32, 32)
    p = ContourletParams(J = 2, L_array = [2, 3])
    c = ct_forward(x, p; threading = Enabled())
    rec = ct_inverse(c, p; threading = Enabled())
    @test maximum(abs, rec .- x) < 1.0e-12
end

@testitem "NSCT with threading=Enabled() and Real data" tags = [:nsct] begin
    using Random
    Random.seed!(91)
    x = randn(32, 32)
    p = ContourletParams(J = 2, L_array = [2, 3])
    nc = nsct_forward(x, p; threading = Enabled())
    rec = nsct_inverse(nc, p; threading = Enabled())
    @test maximum(abs, rec .- x) < 1.0e-12
end

@testitem "ct_forward! integer image coerced to workspace eltype" tags = [:ct] begin
    using Random
    Random.seed!(92)
    x_int = rand(1:9, 32, 32)
    p = ContourletParams(J = 2, L_array = [2, 3])
    ws = make_workspace(p, (32, 32))
    coeffs = similar_coefficients(p, (32, 32))
    ct_forward!(coeffs, x_int, p; workspace = ws)
    x_f64 = Float64.(x_int)
    coeffs_ref = similar_coefficients(p, (32, 32))
    ct_forward!(coeffs_ref, x_f64, p; workspace = ws)
    @test coeffs.coarse == coeffs_ref.coarse
    for j in 1:2, k in eachindex(coeffs.subbands[j])
        @test coeffs.subbands[j][k] == coeffs_ref.subbands[j][k]
    end
end

@testitem "nsct_forward! integer image coerced to workspace eltype" tags = [:nsct] begin
    using Random
    Random.seed!(93)
    x_int = rand(1:9, 32, 32)
    p = ContourletParams(J = 2, L_array = [2, 3])
    ws = make_nsct_workspace(p, (32, 32))
    coeffs = similar_nsct_coefficients(p, (32, 32))
    nsct_forward!(coeffs, x_int, p; workspace = ws)
    x_f64 = Float64.(x_int)
    coeffs_ref = similar_nsct_coefficients(p, (32, 32))
    nsct_forward!(coeffs_ref, x_f64, p; workspace = ws)
    @test coeffs.coarse == coeffs_ref.coarse
    for j in 1:2, k in eachindex(coeffs.subbands[j])
        @test coeffs.subbands[j][k] == coeffs_ref.subbands[j][k]
    end
end
