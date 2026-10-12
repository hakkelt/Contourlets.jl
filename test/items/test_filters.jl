@testitem "CDF97 coefficients" tags = [:filters] begin
    # DC gain of analysis LP should be ≈ 1
    @test abs(sum(CDF97.h) - 1.0) < 1.0e-12
    # Synthesis LP sums to ≈ 2 (compensates for factor-2 upsampling in LP stage)
    @test abs(sum(CDF97.g) - 2.0) < 1.0e-12
end

@testitem "Q2345 PR condition" tags = [:filters] begin
    @test check_pr_condition(Q2345)
end

@testitem "Q2345 structure" tags = [:filters] begin
    # Q2345 is the Phoong et al. (1995) ladder pair: 12-tap symmetric lifting filter.
    @test length(Q2345.f_ladder) == 12
    @test Q2345.f_ladder == reverse(Q2345.f_ladder)          # symmetric
    @test Q2345.f_ladder[7] == 0.63                        # pkva12 taps
end

@testitem "Phoong quincunx pair satisfies the quincunx PR conditions" tags = [:filters] begin
    # Do's thesis eqs. 3.9–3.10 for the closed-form pair built from Q2345's β:
    #   ½(G₀H₀ + G₁H₁) = 1 and H₀(ω+π)G₀(ω) + H₁(ω+π)G₁(ω) = 0, π = (π, π).
    v = Contourlets._ladder_half(Q2345.f_ladder)
    b(θ) = Contourlets._ladder_amplitude(v, θ)
    p(w) = b(w[1] + w[2]) * b(w[1] - w[2])
    H(i, w) = i == 0 ? (1 + p(w)) / sqrt(2) : cis(w[1]) * (2 - p(w) - p(w)^2) / sqrt(2)
    G(i, w) = i == 0 ? (2 + p(w) - p(w)^2) / sqrt(2) : cis(-w[1]) * (1 - p(w)) / sqrt(2)
    for w in ([0.3, -1.1], [2.0, 0.7], [-2.9, 3.1], [0.0, 0.0])
        wπ = w .+ π
        @test abs((G(0, w) * H(0, w) + G(1, w) * H(1, w)) / 2 - 1) < 1.0e-12
        @test abs(H(0, wπ) * G(0, w) + H(1, wπ) * G(1, w)) < 1.0e-12
    end
end

@testitem "NSDFB equivalent filters sum to the identity" tags = [:filters] begin
    # Σₖ GₖHₖ = 1 on the whole grid, for full and half trees.
    for l in 1:4, half in (:both, :first, :second)
        b = Contourlets._directional_bank(Float64, (16, 24), l, Q2345.f_ladder; decimated = false, half = half)
        @test maximum(abs, sum(a .* g for (a, g) in zip(b.ana, b.syn)) .- 1) < 1.0e-12
    end
end

@testitem "upsample_filter" tags = [:filters] begin
    h = [1.0, 2.0, 3.0]
    @test upsample_filter(h, 1) == h
    hu = upsample_filter(h, 2)
    @test length(hu) == 5
    @test hu == [1.0, 0.0, 2.0, 0.0, 3.0]
end

@testitem "FilterPair eltype" tags = [:filters] begin
    @test eltype(CDF97) == Float64
    @test eltype(Q2345) == Float64
end

@testitem "upsample_kernel 2D" tags = [:filters] begin
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

@testitem "upsample_filter invalid factor" tags = [:filters] begin
    @test_throws ArgumentError upsample_filter([1.0, 2.0], 0)
end

@testitem "upsample_kernel invalid factor" tags = [:filters] begin
    @test_throws ArgumentError upsample_kernel([1.0 2.0], 0)
end

@testitem "check_pr_condition(FilterPair)" tags = [:filters] begin
    @test check_pr_condition(CDF97)
end

@testitem "FilterPair type promotion" tags = [:filters] begin
    fp = FilterPair([1.0f0, 2.0f0], [1.0, 2.0])   # Float32 vs Float64
    @test eltype(fp) == Float64
end

@testitem "QuincunxFilterPair constructors" tags = [:filters] begin
    qfp = QuincunxFilterPair(Float32.(Q2345.f_ladder))
    @test qfp isa QuincunxFilterPair{Float32}
    @test eltype(qfp) == Float32
    @test qfp.f_ladder == Float32.(Q2345.f_ladder)
    qfp64 = QuincunxFilterPair{Float64}(Float32[0.25, 0.25])
    @test eltype(qfp64.f_ladder) == Float64
    # Any ladder filter gives a perfect-reconstruction pair.
    @test check_pr_condition(QuincunxFilterPair([0.1, 0.4, 0.4, 0.1]))
end

@testitem "ContourletParams show method" tags = [:filters] begin
    p = ContourletParams(J = 3, L_array = [1, 2, 3])
    s = sprint(show, p)
    @test occursin("J=3", s)
    @test occursin("L_array", s)
end

@testitem "ContourletParams invalid L_array length" tags = [:filters] begin
    @test_throws ArgumentError ContourletParams(J = 3, L_array = [1, 2])
end

@testitem "ContourletParams negative L_array entry" tags = [:filters] begin
    @test_throws ArgumentError ContourletParams(J = 2, L_array = [1, -1])
end

@testitem "pkva_ladder_filter symmetric 12-tap" tags = [:filters] begin
    f = Contourlets.pkva_ladder_filter()
    @test length(f) == 12
    @test f == reverse(f)   # symmetric
    @test f[6] == 0.63       # pkva12 tap check (f[6] = _PKVA12_HALF[1])
end

@testitem "ContourletParams eltype and same-type conversions are identities" tags = [:filters] begin
    using Random
    p = ContourletParams(J = 2, L_array = [1, 2])
    @test eltype(p) == Float64
    @test Contourlets._convert_params(Float64, p) === p
    @test Contourlets._convert_qfp(Float64, Q2345) === Q2345
    p32 = Contourlets._convert_params(Float32, p)
    @test eltype(p32) == Float32
    @test eltype(p32.dfb_filters) == Float32
    @test Contourlets._convert_params(Float32, p32) === p32
end

@testitem "Coefficient containers infer type parameters from 2-arg constructors" tags = [:filters] begin
    c = zeros(4, 4)
    sbs = [[zeros(4, 4), zeros(4, 4)]]
    cc = ContourletCoefficients(c, sbs)
    @test cc isa ContourletCoefficients{Float64, Matrix{Float64}}
    @test cc.coarse === c
    @test cc.subbands === sbs
    zc = zeros(ComplexF32, 2, 2)
    nc = NSCTCoefficients(zc, [[zeros(ComplexF32, 2, 2)]])
    @test nc isa NSCTCoefficients{ComplexF32, Matrix{ComplexF32}}
end
