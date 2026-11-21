# Complex-data support: the transforms are linear with real filters, so a complex
# image is filtered as complex·real (filters are never promoted to complex) and
#   T(x + i·y) ≈ T(x) + i·T(y).
# Complex data runs through complex FFTs and real data through real ones, so the
# two sides agree to rounding, not bit for bit.

@testitem "CT complex round-trip and eltype" tags = [:ct] begin
    using Random
    Random.seed!(40)
    t = CT(scales = 3, direction_levels = parabolic_levels(3))
    z = complex.(randn(64, 64), randn(64, 64))
    c = analyze(t, z)
    @test eltype(c) == ComplexF64
    rec = synthesize(t, c)
    @test eltype(rec) == ComplexF64
    @test maximum(abs, rec .- z) < 1.0e-12
end

@testitem "complex linearity (real filters)" tags = [:ct, :nsct, :tict] begin
    using Random
    Random.seed!(41)
    x = randn(64, 64)
    y = randn(64, 64)
    for t in (CT(scales = 2, direction_levels = [2, 3]), NSCT(scales = 2, direction_levels = [2, 3]), TICT(scales = 2, direction_levels = [2, 3]), STICT(scales = 2, direction_levels = [2, 3]))
        cz = analyze(t, complex.(x, y))
        @test cz ≈ complex.(analyze(t, x), analyze(t, y))
    end
end

@testitem "NSCT complex round-trip and shift-invariance" tags = [:nsct] begin
    using Random
    Random.seed!(42)
    t = NSCT(scales = 2, direction_levels = [2, 3])
    z = complex.(randn(48, 48), randn(48, 48))
    c = analyze(t, z)
    @test eltype(c) == ComplexF64
    rec = synthesize(t, c)
    @test maximum(abs, rec .- z) < 1.0e-12
    # circular-shift invariance carries over to complex data
    s = (5, 3)
    cs = analyze(t, circshift(z, s))
    for i in 2:size(c, 3)
        @test maximum(abs, subband(t, cs, i) .- circshift(subband(t, c, i), s)) < 1.0e-10
    end
end

@testitem "ComplexF32 precision is preserved" tags = [:ct, :nsct] begin
    using Random
    Random.seed!(43)
    z = ComplexF32.(complex.(randn(Float32, 64, 64), randn(Float32, 64, 64)))
    for t in (CT(scales = 2, direction_levels = [2, 3]), NSCT(scales = 2, direction_levels = [2, 3]))
        c = analyze(t, z)
        @test eltype(c) == ComplexF32
        @test maximum(abs, synthesize(t, c) .- z) < 1.0f-4
    end
end

@testitem "complex workspace path matches allocating" tags = [:ct, :nsct] begin
    using Random
    Random.seed!(44)
    z = complex.(randn(64, 64), randn(64, 64))
    t = CT(scales = 2, direction_levels = [1, 2])
    ws = make_workspace(t, ComplexF64, (64, 64))
    @test ws isa CTWorkspace{ComplexF64, Float64}
    cw = analyze(t, z; workspace = ws)
    @test subband(t, cw, 1) ≈ subband(t, analyze(t, z), 1)
    @test maximum(abs, synthesize(t, cw) .- z) < 1.0e-12

    tn = NSCT(scales = 2, direction_levels = [1, 2])
    wsn = make_workspace(tn, ComplexF64, (64, 64))
    nw = analyze(tn, z; workspace = wsn)
    @test subband(tn, nw, 1) ≈ subband(tn, analyze(tn, z), 1)
end

@testitem "primitive conv2d_sep with complex data, real filter" tags = [:primitives] begin
    using Random
    Random.seed!(45)
    h = Float64[0.25, 0.5, 0.25]
    x = randn(16, 16)
    y = randn(16, 16)
    rz = Contourlets.conv2d_sep(complex.(x, y), h, h)
    @test eltype(rz) == ComplexF64
    @test real(rz) ≈ Contourlets.conv2d_sep(x, h, h)
    @test imag(rz) ≈ Contourlets.conv2d_sep(y, h, h)
end
