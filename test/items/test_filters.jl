@testitem "CDF97 coefficients" begin
    # DC gain of analysis LP should be ≈ 1
    @test abs(sum(CDF97.h) - 1.0) < 1.0e-12
    # Synthesis LP sums to ≈ 2 (compensates for factor-2 upsampling in LP stage)
    @test abs(sum(CDF97.g) - 2.0) < 1.0e-12
end

@testitem "Q2345 PR condition" begin
    @test check_pr_condition(Q2345)
end

@testitem "Q2345 structure" begin
    # Q2345 is the Phoong et al. (1995) ladder pair: 12-tap symmetric lifting
    # filter, with 23-tap / 45-tap equivalent analysis/synthesis low-pass filters.
    @test Contourlets.is_ladder(Q2345)
    @test length(Q2345.f_ladder) == 12
    @test Q2345.f_ladder == reverse(Q2345.f_ladder)          # symmetric
    @test Q2345.f_ladder[7] == 0.63                        # pkva12 taps
    # h_q and g_q hold the equivalent 1-row filters ("23-45")
    @test size(Q2345.h_q) == (1, 23)
    @test size(Q2345.g_q) == (1, 45)
end

@testitem "Modulation-mode Haar pair PR" begin
    using Random
    Random.seed!(11)
    haar = QuincunxFilterPair([0.5 0.5], [1.0 1.0], (1, 1), (1, 2))
    @test !Contourlets.is_ladder(haar)
    @test check_pr_condition(haar)
    x = randn(32, 32)
    sbs = dfb_decompose(x, 3, haar)
    @test maximum(abs, dfb_reconstruct(sbs, haar) .- x) < 1.0e-12
    nsbs = nsdfb_decompose(x, 2, haar, 2)
    @test maximum(abs, nsdfb_reconstruct(nsbs, haar, 2) .- x) < 1.0e-12
end

@testitem "Q2345 NSDFB equivalent-filter PR identity" begin
    # 0.5·(G₀H₀ + G₁H₁) = δ exactly (zero delay) — the nonsubsampled PR identity
    qup = Contourlets._upsample_qfp_1d(Q2345, 1, Float64)
    a, ca = Contourlets._lag_conv(qup.g0, qup.cg0, qup.h0, qup.c0)
    b, cb = Contourlets._lag_conv(qup.g1, qup.cg1, qup.h1, qup.c1)
    c = max(ca, cb)
    n = c + max(length(a) - ca, length(b) - cb)
    p = zeros(n)
    p[(c - ca + 1):(c - ca + length(a))] .+= a
    p[(c - cb + 1):(c - cb + length(b))] .+= b
    p .*= qup.scale
    @test abs(p[c] - 1) < 1.0e-14
    p[c] = 0
    @test maximum(abs, p) < 1.0e-14
end

@testitem "upsample_filter" begin
    h = [1.0, 2.0, 3.0]
    @test upsample_filter(h, 1) == h
    hu = upsample_filter(h, 2)
    @test length(hu) == 5
    @test hu == [1.0, 0.0, 2.0, 0.0, 3.0]
end

@testitem "FilterPair eltype" begin
    @test eltype(CDF97) == Float64
    @test eltype(Q2345) == Float64
end

@testitem "upsample_kernel 2D" begin
    K = [1.0 2.0; 3.0 4.0]
    K2 = upsample_kernel(K, 1)
    @test K2 == K
    K3 = upsample_kernel(K, 2)
    @test size(K3) == (3, 3)
    @test K3[1, 1] == 1.0
    @test K3[1, 3] == 2.0
    @test K3[3, 1] == 3.0
    @test K3[3, 3] == 4.0
    @test K3[1, 2] == 0.0
end

@testitem "upsample_filter invalid factor" begin
    @test_throws ArgumentError upsample_filter([1.0, 2.0], 0)
end

@testitem "upsample_kernel invalid factor" begin
    @test_throws ArgumentError upsample_kernel([1.0 2.0], 0)
end

@testitem "check_pr_condition(FilterPair)" begin
    @test check_pr_condition(CDF97)
end

@testitem "FilterPair type promotion" begin
    fp = FilterPair([1.0f0, 2.0f0], [1.0, 2.0])   # Float32 vs Float64
    @test eltype(fp) == Float64
end

@testitem "QuincunxFilterPair generic constructor" begin
    qfp = QuincunxFilterPair([0.5 0.5], [1.0 1.0], (1, 1), (1, 2))
    @test eltype(qfp) == Float64
    @test qfp.c_h == (1, 1)
    @test qfp.c_g == (1, 2)
end

@testitem "ContourletParams show method" begin
    p = ContourletParams(J = 3, L_array = [1, 2, 3])
    s = sprint(show, p)
    @test occursin("J=3", s)
    @test occursin("L_array", s)
end

@testitem "ContourletParams invalid L_array length" begin
    @test_throws ArgumentError ContourletParams(J = 3, L_array = [1, 2])
end

@testitem "ContourletParams negative L_array entry" begin
    @test_throws ArgumentError ContourletParams(J = 2, L_array = [1, -1])
end
