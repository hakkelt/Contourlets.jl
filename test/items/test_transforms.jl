@testitem "CT analyze/synthesize PR" tags = [:ct] begin
    using Random
    Random.seed!(30)
    x = randn(32, 32)
    t = CT(scales = 2, direction_levels = [3, 2])
    c = analyze(t, x)
    @test c isa SubbandVector{Float64}
    rec = synthesize(t, c)
    @test maximum(abs, rec .- x) < 1.0e-12
end

@testitem "CT coefficient structure" tags = [:ct] begin
    x = zeros(32, 32); x[16, 16] = 1.0
    t = CT(scales = 2, direction_levels = [2, 3])
    c = analyze(t, x)
    @test nsubbands(t, size(x)) == 1 + 4 + 8
    layout = subband_layout(t, size(x))
    @test layout[1] == (scale = 0, direction = 0)
    @test count(s -> s.scale == 1, layout) == 4    # 2^2 directional subbands at the coarsest scale
    @test count(s -> s.scale == 2, layout) == 8    # 2^3 directional subbands at the finest scale
    @test [size(subband(t, c, i)) for i in (2, 6)] == [(8, 8), (8, 16)]   # first of each scale
    @test size(subband(t, c, 1)) == (8, 8)
    @test sum(i -> length(subband(t, c, i)), 1:13) == length(c)
end

@testitem "CT Float32 round-trip" tags = [:ct] begin
    using Random
    Random.seed!(31)
    x = randn(Float32, 32, 32)
    t = CT(scales = 1, direction_levels = [2])
    c = analyze(t, x)
    @test eltype(c) == Float32
    rec = synthesize(t, c)
    @test eltype(rec) == Float32
    @test maximum(abs, rec .- x) < 1.0e-4
end

@testitem "NSCT analyze/synthesize PR" tags = [:nsct] begin
    using Random
    Random.seed!(32)
    x = randn(32, 32)
    t = NSCT(scales = 2, direction_levels = [3, 2])
    nc = analyze(t, x)
    @test nc isa Array{Float64, 3}
    @test size(nc) == (32, 32, 13)
    rec = synthesize(t, nc)
    @test maximum(abs, rec .- x) < 1.0e-12
end

@testitem "NSCT shift invariance" tags = [:nsct] begin
    using Random
    Random.seed!(38)
    x = randn(32, 32)
    t = NSCT(scales = 2, direction_levels = [3, 2])
    c0 = analyze(t, x)
    for s in ((1, 0), (0, 1), (3, 5), (7, 11))
        cs = analyze(t, circshift(x, s))
        # Coefficients of the shifted image equal the shifted coefficients,
        # with no redistribution of energy across subbands.
        for i in axes(c0, 3)
            @test maximum(abs, subband(t, cs, i) .- circshift(subband(t, c0, i), s)) < 1.0e-12
        end
    end
end

@testitem "NSCT subband energy is shift invariant" tags = [:nsct] begin
    using Random
    Random.seed!(39)
    x = randn(32, 32)
    t = NSCT(scales = 2, direction_levels = [3, 2])
    c0, cs = analyze(t, x), analyze(t, circshift(x, (4, 9)))
    e0 = [sum(abs2, subband(t, c0, i)) for i in 2:size(c0, 3)]
    es = [sum(abs2, subband(t, cs, i)) for i in 2:size(cs, 3)]
    @test maximum(abs, e0 .- es) < 1.0e-8
end

@testitem "NSCT all subbands same size as input" tags = [:nsct] begin
    x = zeros(32, 32)
    t = NSCT(scales = 2, direction_levels = [3, 2])
    nc = analyze(t, x)
    @test all(size(subband(t, nc, i)) == size(x) for i in 1:nsubbands(t, size(x)))
end

@testitem "transform constructors validate their parameters" tags = [:ct] begin
    for T in (CT, NSCT, TICT, STICT)
        @test_throws "scales must be ≥ 1" T(scales = 0, direction_levels = Int[])
        @test_throws "scales must be ≥ 1" T(scales = 0)
        @test_throws "scales must be ≥ 1" T(direction_levels = Int[])
        @test_throws "scales = 2 disagrees with direction_levels = [1]" T(scales = 2, direction_levels = [1])
        @test_throws "direction_levels entries must be ≥ 0" T(scales = 2, direction_levels = [1, -1])
        @test_throws "give scales, direction_levels, or both" T()
        @test_throws "length(direction_levels) must equal scales = 2" T{Float64}(2, [1], CDF97, Q2345)
    end
end

@testitem "transform constructors complete scales and direction_levels" tags = [:ct] begin
    for T in (CT, NSCT, TICT, STICT)
        @test T(scales = 3) == T(scales = 3, direction_levels = parabolic_levels(3))
        @test T(direction_levels = [1, 1, 2]) == T(scales = 3, direction_levels = [1, 1, 2])
        @test T(scales = Int32(2), direction_levels = UInt8[1, 2]) == T(scales = 2, direction_levels = [1, 2])
        @test T(scales = 2).scales === 2
    end
end

@testitem "transform constructor forms agree" tags = [:ct] begin
    lp32 = FilterPair(Float32.(CDF97.h), Float32.(CDF97.g))
    q32 = QuincunxFilterPair(Float32.(Q2345.f_ladder))
    for T in (CT, NSCT, TICT, STICT)
        a = T(scales = 2, direction_levels = [1, 2])
        @test a isa T{Float64}
        @test T{Float64}(scales = 2, direction_levels = [1, 2]) == a
        @test T{Float64}(2, [1, 2], CDF97, Q2345) == a
        @test T{Float64}(direction_levels = [1, 2]) == a
        @test T{Float64}(scales = Int32(2)) == T(scales = 2)
        b = T{Float32}(scales = 2, direction_levels = [1, 2])
        @test b isa T{Float32}
        @test b == T(scales = 2, direction_levels = [1, 2], lp_filters = lp32, dfb_filters = q32)
        @test b == T{Float32}(2, [1, 2], CDF97, Q2345)
        @test b == Contourlets._convert_transform(Float32, a)
        @test T{Float64}(scales = 2, lp_filters = lp32, dfb_filters = q32) isa T{Float64}
    end
end

@testitem "QuincunxFilterPair constructor forms agree" tags = [:filters] begin
    q = QuincunxFilterPair([1, 2, 2, 1])
    @test q isa QuincunxFilterPair{Float64}
    @test q == QuincunxFilterPair([1.0, 2.0, 2.0, 1.0]) == QuincunxFilterPair{Float64}([1, 2, 2, 1])
    @test QuincunxFilterPair{Float32}([1, 2, 2, 1]) == QuincunxFilterPair(Float32[1, 2, 2, 1])
    f = [0.25, 0.25]
    qf = QuincunxFilterPair(f)
    f[1] = 1.0
    @test qf.f_ladder == [0.25, 0.25]    # the pair holds a copy
end

@testitem "transforms compare and hash by value" tags = [:ct] begin
    for T in (CT, NSCT, TICT, STICT)
        a = T(scales = 2, direction_levels = [2, 1])
        b = T(
            scales = 2, direction_levels = [2, 1], lp_filters = FilterPair(collect(CDF97.h), collect(CDF97.g)),
            dfb_filters = QuincunxFilterPair(copy(Q2345.f_ladder))
        )
        @test a == b
        @test hash(a) == hash(b)
        @test a != T(scales = 2, direction_levels = [2, 2])
        @test a != T(scales = 2, direction_levels = [2, 1], dfb_filters = QuincunxFilterPair(2 .* Q2345.f_ladder))
        @test isequal(a, b)
        @test length(Set([a, b])) == 1
    end
    @test CT(scales = 2, direction_levels = [2, 1]) != NSCT(scales = 2, direction_levels = [2, 1])
    @test QuincunxFilterPair(copy(Q2345.f_ladder)) == Q2345
    @test hash(QuincunxFilterPair(copy(Q2345.f_ladder))) == hash(Q2345)
    # The keyword constructor copies direction_levels.
    L = [1, 2]
    t = CT(scales = 2, direction_levels = L)
    L[1] = 5
    @test t.direction_levels == [1, 2]
end

@testitem "transform show" tags = [:ct] begin
    @test repr(CT(scales = 2, direction_levels = [1, 2])) == "CT(scales = 2, direction_levels = [1, 2])"
    @test repr(TICT(scales = 1, direction_levels = [3])) == "TICT(scales = 1, direction_levels = [3])"
    s = repr(NSCT(scales = 1, direction_levels = [1], dfb_filters = QuincunxFilterPair(2 .* Q2345.f_ladder)))
    @test startswith(s, "NSCT(scales = 1, direction_levels = [1], dfb_filters = ")
end

@testitem "parabolic_levels helper" tags = [:ct] begin
    ls = parabolic_levels(4)
    @test length(ls) == 4
    @test all(x -> x >= 1, ls)
    @test issorted(ls)  # finer scales have more directions
    @test ls == [1, 1, 2, 2]
end

@testitem "CT analyze!/synthesize! with workspace" tags = [:ct] begin
    using Random
    Random.seed!(34)
    x = randn(32, 32)
    t = CT(scales = 2, direction_levels = [3, 2])
    ws = make_workspace(t, Float64, (32, 32))
    coeffs = similar_coefficients(t, x)
    @test analyze!(coeffs, t, x; workspace = ws) === coeffs
    rec = similar(x)
    @test synthesize!(rec, t, coeffs; workspace = ws) === rec
    @test maximum(abs, rec .- x) < 1.0e-12
end

@testitem "CT workspace analysis matches the allocating one" tags = [:ct] begin
    using Random
    Random.seed!(35)
    x = randn(32, 32)
    t = CT(scales = 2, direction_levels = [3, 2])
    ws = make_workspace(t, Float64, (32, 32))
    coeffs = similar_coefficients(t, x)
    analyze!(coeffs, t, x; workspace = ws)
    coeffs_alloc = analyze(t, x)
    @test maximum(abs, coeffs .- coeffs_alloc) < 1.0e-12
end

@testitem "NSCT analyze!/synthesize! with workspace" tags = [:nsct] begin
    using Random
    Random.seed!(36)
    x = randn(32, 32)
    t = NSCT(scales = 2, direction_levels = [3, 2])
    ws = make_workspace(t, Float64, (32, 32))
    coeffs = similar_coefficients(t, x)
    analyze!(coeffs, t, x; workspace = ws)
    rec = similar(x)
    synthesize!(rec, t, coeffs; workspace = ws)
    @test maximum(abs, rec .- x) < 1.0e-12
end

@testitem "NSCT workspace analysis matches the allocating one" tags = [:nsct] begin
    using Random
    Random.seed!(37)
    x = randn(32, 32)
    t = NSCT(scales = 2, direction_levels = [3, 2])
    ws = make_workspace(t, Float64, (32, 32))
    coeffs = similar_coefficients(t, x)
    analyze!(coeffs, t, x; workspace = ws)
    coeffs_alloc = analyze(t, x)
    @test maximum(abs, subband(t, coeffs, 1) .- subband(t, coeffs_alloc, 1)) < 1.0e-12
    @test isapprox(coeffs, coeffs_alloc; atol = 1.0e-10)
end

@testitem "similar_coefficients sizes" tags = [:ct, :nsct] begin
    x = zeros(64, 64)
    t = NSCT(scales = 2, direction_levels = [3, 2])
    nc = similar_coefficients(t, x)
    @test size(nc) == (64, 64, 1 + 4 + 8)
    @test iszero(nc)
    @test similar_coefficients(t, zeros(Float32, 64, 64)) isa Array{Float32, 3}
    @test similar_coefficients(t, zeros(Int, 64, 64)) isa Array{Float64, 3}
    ct = CT(scales = 2, direction_levels = [3, 2])
    c = similar_coefficients(ct, zeros(32, 32))
    @test c isa SubbandVector{Float64}
    @test nsubbands(c) == 13
    @test size(subband(ct, c, 1)) == (8, 8)
    @test iszero(c)
end

@testitem "parabolic_levels(0) returns empty" tags = [:ct] begin
    ls = parabolic_levels(0)
    @test ls == Int[]
    ls1 = parabolic_levels(0, 2)
    @test ls1 == Int[]
end

@testitem "parabolic_levels with a deeper coarsest scale" tags = [:ct] begin
    ls = parabolic_levels(5, 2)
    @test ls == [2, 2, 3, 3, 4]   # coarsest first
end

@testitem "threaded keyword resolution" tags = [:ct] begin
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        @test Contourlets.use_threading(true, T) === true
        @test Contourlets.use_threading(false, T) === false
    end
    @test Contourlets.use_threading(nothing, Float64) === false
    @test Contourlets.use_threading(nothing, ComplexF64) === true
    @test Contourlets.use_threading(nothing, Float32) === false
    @test Contourlets.use_threading(nothing, ComplexF32) === true
end

@testitem "threaded keyword is accepted on every path" tags = [:ct, :nsct, :tict] begin
    using Random
    Random.seed!(91)
    x = randn(32, 32)
    for t in (CT(scales = 2, direction_levels = [3, 2]), NSCT(scales = 2, direction_levels = [3, 2]), TICT(scales = 2, direction_levels = [3, 2]), STICT(scales = 2, direction_levels = [3, 2]))
        for threaded in (nothing, true, false)
            c = analyze(t, x; threaded)
            @test maximum(abs, synthesize(t, c; threaded) .- x) < 1.0e-12
            ws = make_workspace(t, x)
            @test synthesize(ws, analyze(ws, x; threaded); threaded) ≈ x atol = 1.0e-12
        end
    end
end

@testitem "integer image coerced to the workspace element type" tags = [:ct, :nsct] begin
    using Random
    Random.seed!(92)
    x_int = rand(1:9, 32, 32)
    for t in (CT(scales = 2, direction_levels = [3, 2]), NSCT(scales = 2, direction_levels = [3, 2]))
        ws = make_workspace(t, Float64, (32, 32))
        @test eltype(ws) == Float64
        @test make_workspace(t, x_int) isa typeof(ws)
        coeffs = similar_coefficients(t, x_int)
        analyze!(coeffs, t, x_int; workspace = ws)
        coeffs_ref = similar_coefficients(t, x_int)
        analyze!(coeffs_ref, t, Float64.(x_int); workspace = ws)
        @test coeffs == coeffs_ref
    end
end

@testitem "analyze! without a workspace matches analyze" tags = [:ct, :nsct] begin
    using Random
    Random.seed!(42)
    x = randn(32, 32)
    for t in (CT(scales = 2, direction_levels = [3, 2]), NSCT(scales = 2, direction_levels = [3, 2]))
        ref = analyze(t, x)
        coeffs = similar_coefficients(t, x)
        @test analyze!(coeffs, t, x) === coeffs
        @test coeffs == ref
        # An integer image is analyzed as Float64.
        xi = rand(Xoshiro(42), 0:9, 32, 32)
        analyze!(coeffs, t, xi)
        @test coeffs == analyze(t, Float64.(xi))
        # Complex coefficients, real-filtered.
        z = complex.(randn(32, 32), randn(32, 32))
        cz = similar_coefficients(t, z)
        @test eltype(cz) == ComplexF64
        analyze!(cz, t, z)
        @test cz == analyze(t, z)
    end
end

@testitem "allocating synthesis with a workspace matches the one without" tags = [:ct, :nsct] begin
    using Random
    Random.seed!(42)
    x = randn(32, 32)
    for t in (CT(scales = 2, direction_levels = [3, 2]), NSCT(scales = 2, direction_levels = [3, 2]))
        ws = make_workspace(t, Float64, (32, 32))
        c = analyze(t, x)
        rec_ws = synthesize(t, c; workspace = ws)
        rec = synthesize(t, c)
        @test rec_ws isa Matrix{Float64}
        @test size(rec_ws) == size(x)
        @test maximum(abs, rec_ws .- rec) < 1.0e-10
        @test maximum(abs, rec_ws .- x) < 1.0e-10
        @test synthesize(ws, c) == rec_ws
    end
end

@testitem "wrong arguments fail fast" tags = [:ct, :nsct, :tict] begin
    x = randn(32, 32)
    for t in (CT(scales = 2, direction_levels = [3, 2]), NSCT(scales = 2, direction_levels = [3, 2]), TICT(scales = 2, direction_levels = [3, 2]), STICT(scales = 2, direction_levels = [3, 2]))
        @test_throws "the contourlet transforms take images (two axes)" analyze(t, randn(8, 8, 8))
        @test_throws "the contourlet transforms take images (two axes)" make_workspace(t, Float64, (8, 8, 8))
        @test_throws "the contourlet transforms take images (two axes)" subband_layout(t, (8,))
        ws = make_workspace(t, Float64, (32, 32))
        # Another size, element type or transform.
        @test_throws DimensionMismatch analyze(t, randn(64, 64); workspace = ws)
        @test_throws "the workspace was built for element type Float64, not for Float32" analyze(t, randn(Float32, 32, 32); workspace = ws)
        other = make_workspace(t isa CT ? NSCT(scales = 2, direction_levels = [3, 2]) : CT(scales = 2, direction_levels = [3, 2]), Float64, (32, 32))
        @test_throws "the workspace was built for" analyze(t, x; workspace = other)
        # Coefficients of another size or element type.
        c = similar_coefficients(t, randn(64, 64))
        @test_throws DimensionMismatch analyze!(c, t, x)
        @test_throws DimensionMismatch analyze!(c, t, x; workspace = ws)
        @test_throws DimensionMismatch synthesize!(similar(x), t, c)
        c32 = similar_coefficients(t, randn(Float32, 32, 32))
        @test_throws "element type Float32, but the data needs Float64" analyze!(c32, t, x)
        @test_throws "element type Float32, but the data needs Float64" analyze!(c32, t, x; workspace = ws)
        @test_throws "element type Float32, but the data needs Float64" synthesize!(similar(x), t, c32; workspace = ws)
    end
end

@testitem "workspace accessors" tags = [:ct, :nsct, :tict] begin
    for t in (CT(scales = 2, direction_levels = [3, 2]), NSCT(scales = 2, direction_levels = [3, 2]), TICT(scales = 2, direction_levels = [3, 2]), STICT(scales = 2, direction_levels = [3, 2]))
        ws = make_workspace(t, ComplexF32, (32, 32))
        @test ws isa AbstractTransformWorkspace
        @test transform(ws) === t
        @test eltype(ws) == ComplexF32
        @test size(ws) == (32, 32)
        @test eltype(make_workspace(t, Int, (32, 32))) == Float64
        @test occursin(string(t), repr(ws))
    end
end

@testitem "stacked coefficients of another array type go through a copy" tags = [:nsct, :tict] begin
    using Random
    Random.seed!(95)
    x = randn(32, 32)
    for t in (NSCT(scales = 2, direction_levels = [3, 2]), TICT(scales = 2, direction_levels = [2, 1]))
        ws = make_workspace(t, x)
        ref = analyze(t, x; workspace = ws)
        big = zeros(32, 32, size(ref, 3) + 1)
        v = view(big, :, :, 2:size(big, 3))
        @test analyze!(v, t, x; workspace = ws) === v
        @test v == ref
        @test synthesize!(similar(x), t, v; workspace = ws) ≈ x atol = 1.0e-12
    end
end

@testitem "NSCT and TICT on odd sizes (unaligned subband slices)" tags = [:nsct, :tict] begin
    using Random
    Random.seed!(96)
    for t in (NSCT(scales = 2, direction_levels = [2, 1]), TICT(scales = 2, direction_levels = [1, 2])), T in (Float64, Float32, ComplexF32)
        x = randn(T, 31, 33)
        ws = make_workspace(t, x)
        c = analyze(t, x; workspace = ws)
        @test c ≈ analyze(t, x) rtol = sqrt(eps(real(T)))
        @test synthesize(t, c; workspace = ws) ≈ x rtol = sqrt(eps(real(T)))
    end
end

@testitem "transforms print their element type and round-trip" tags = [:ct] begin
    for F in (CT, NSCT, TICT, STICT), T in (Float32, Float64)
        t = F{T}(scales = 2, direction_levels = [1, 2])
        s = repr(t)
        @test occursin("{Float32}", s) == (T === Float32)
        @test eval(Meta.parse(s)) == t
    end
    # A precision mismatch now reads differently on both sides.
    ws = make_workspace(CT{Float32}(scales = 2), Float32, (32, 32))
    @test_throws "built for CT{Float32}(scales = 2" analyze(CT(scales = 2), randn(Float32, 32, 32); workspace = ws)
end

@testitem "stacked coefficients of the wrong size are rejected without a workspace" tags = [:nsct, :tict] begin
    for t in (NSCT(scales = 2, direction_levels = [2, 1]), TICT(scales = 2, direction_levels = [2, 1]))
        x = randn(32, 32)
        c = similar_coefficients(t, randn(64, 64))
        @test_throws "do not fit an image of size (32, 32)" synthesize!(similar(x), t, c)
        @test_throws "do not fit an image of size (32, 32)" analyze!(c, t, x)
    end
end

@testitem "STICT direct synthesis converts to the output's element type" tags = [:tict] begin
    using Random
    Random.seed!(77)
    t = STICT(scales = 2, direction_levels = [2, 1])
    x = randn(Float32, 32, 32)
    c = analyze(t, x)
    @test eltype(c) == Float32
    out = zeros(Float64, 32, 32)
    @test synthesize!(out, t, c) === out
    @test maximum(abs, out .- x) < 1.0e-4
end
