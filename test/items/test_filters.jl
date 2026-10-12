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

@testitem "Modulation-mode Haar pair PR" tags = [:filters] begin
    using Random
    Random.seed!(11)
    haar = QuincunxFilterPair([0.5 0.5], [1.0 1.0], (1, 1), (1, 2))
    @test !Contourlets.is_ladder(haar)
    @test check_pr_condition(haar)
    x = randn(32, 32)
    sbs = dfb_decompose(x, 3, haar)
    @test maximum(abs, dfb_reconstruct(sbs, haar) .- x) < 1.0e-12
    # The resampling-matrix NSDFB needs diamond (pkva ladder) filters; haar is not
    # a directional/diamond filter, so it is rejected with a clear error.
    @test_throws ArgumentError nsdfb_decompose(x, 2, haar, 2)
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

@testitem "QuincunxFilterPair generic constructor" tags = [:filters] begin
    qfp = QuincunxFilterPair([0.5 0.5], [1.0 1.0], (1, 1), (1, 2))
    @test eltype(qfp) == Float64
    @test qfp.c_h == (1, 1)
    @test qfp.c_g == (1, 2)
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

@testitem "QuincunxFilterPair{T} 4-arg modulation-mode constructor" tags = [:filters] begin
    qfp = QuincunxFilterPair{Float32}([0.5 0.5], [1.0 1.0], (1, 1), (1, 2))
    @test qfp isa QuincunxFilterPair{Float32}
    @test eltype(qfp) == Float32
    @test qfp.h_q == Float32[0.5 0.5]
    @test qfp.g_q == Float32[1.0 1.0]
    @test isempty(qfp.f_ladder)
    @test eltype(qfp.f_ladder) == Float32
    @test !Contourlets.is_ladder(qfp)
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

@testitem "_ladder_equivalent_filters reproduces Q2345 and satisfies PR" tags = [:filters] begin
    f = Contourlets._ladder_modulate(Contourlets.pkva_ladder_filter())
    eq = Contourlets._ladder_equivalent_filters(f)
    # Stored Q2345 equivalent filters are exactly these (computed at load time).
    @test eq.h0 == vec(Q2345.h_q)
    @test eq.g0 == vec(Q2345.g_q)
    @test (1, eq.c0) == Q2345.c_h
    @test (1, eq.cg0) == Q2345.c_g
    @test length(eq.h0) == 23
    @test length(eq.g0) == 45
    # PR identity (g0 ⊛ h0 + g1 ⊛ h1) / 2 = δ at zero lag.
    conv(a, ca, b, cb) = Contourlets._lag_conv(a, ca, b, cb)
    p0, c_p0 = conv(eq.g0, eq.cg0, eq.h0, eq.c0)
    p1, c_p1 = conv(eq.g1, eq.cg1, eq.h1, eq.c1)
    lo = min(1 - c_p0, 1 - c_p1)
    hi = max(length(p0) - c_p0, length(p1) - c_p1)
    acc = zeros(hi - lo + 1)
    for (p, c) in ((p0, c_p0), (p1, c_p1)), m in eachindex(p)
        acc[m - c - lo + 1] += p[m] / 2
    end
    δ = zeros(length(acc))
    δ[1 - lo] = 1
    @test maximum(abs, acc .- δ) < 1.0e-12
    # Works for any element type / lifting filter (PR is structural).
    eq32 = Contourlets._ladder_equivalent_filters(Float32[0.25, -0.5, 0.25])
    @test eltype(eq32.h0) == Float32
    @test eltype(eq32.g1) == Float32
end
