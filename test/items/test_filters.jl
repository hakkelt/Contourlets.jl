@testitem "CDF97 coefficients" begin
    using Contourlets
    # DC gain of analysis LP should be ≈ 1
    @test abs(sum(CDF97.h) - 1.0) < 1.0e-12
    # Synthesis LP sums to ≈ 2 (compensates for factor-2 upsampling in LP stage)
    @test abs(sum(CDF97.g) - 2.0) < 1.0e-12
end

@testitem "Q2345 PR condition" begin
    using Contourlets
    @test check_pr_condition(Q2345)
end

@testitem "Q2345 structure" begin
    using Contourlets
    # h_q and g_q are 1-row (column-direction) filters
    @test size(Q2345.h_q, 1) == 1
    @test size(Q2345.g_q, 1) == 1
    @test Q2345.c_h == (1, 1)
    @test Q2345.c_g == (1, 2)
end

@testitem "upsample_filter" begin
    using Contourlets
    h = [1.0, 2.0, 3.0]
    @test upsample_filter(h, 1) == h
    hu = upsample_filter(h, 2)
    @test length(hu) == 5
    @test hu == [1.0, 0.0, 2.0, 0.0, 3.0]
end

@testitem "FilterPair eltype" begin
    using Contourlets
    @test eltype(CDF97) == Float64
    @test eltype(Q2345) == Float64
end
