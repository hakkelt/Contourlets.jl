# Complex-data support: the transforms are linear with real filters, so a complex
# image is filtered as complex·real (filters are never promoted to complex) and
#   T(x + i·y) == T(x) + i·T(y)
# holds bit-for-bit.

@testitem "CT complex round-trip and eltype" tags = [:ct] begin
    using Random
    Random.seed!(40)
    p = ContourletParams(J = 3, L_array = parabolic_levels(3))
    z = complex.(randn(64, 64), randn(64, 64))
    c = ct_forward(z, p)
    @test eltype(c.coarse) == ComplexF64
    @test all(eltype(sb) == ComplexF64 for level in c.subbands for sb in level)
    @test eltype(c.params) == Float64           # filters stay real
    rec = ct_inverse(c)
    @test eltype(rec) == ComplexF64
    @test maximum(abs, rec .- z) < 1.0e-12
end

@testitem "CT complex linearity (real filters)" tags = [:ct] begin
    using Random
    Random.seed!(41)
    p = ContourletParams(J = 2, L_array = [2, 3])
    x = randn(64, 64)
    y = randn(64, 64)
    cz = ct_forward(complex.(x, y), p)
    cx = ct_forward(x, p)
    cy = ct_forward(y, p)
    # Approximate equality: a real-filtered complex transform splits over re/im parts.
    # Floating-point ordering differences due to LoopVectorization on the real path
    # mean this is no longer bit-for-bit identical, but it is mathematically linear.
    @test cz.coarse ≈ cx.coarse .+ im .* cy.coarse
    for j in 1:p.J, k in eachindex(cz.subbands[j])
        @test cz.subbands[j][k] ≈ cx.subbands[j][k] .+ im .* cy.subbands[j][k]
    end
end

@testitem "NSCT complex round-trip and shift-invariance" tags = [:nsct] begin
    using Random
    Random.seed!(42)
    p = ContourletParams(J = 2, L_array = [2, 3])
    z = complex.(randn(48, 48), randn(48, 48))
    c = nsct_forward(z, p)
    @test eltype(c.coarse) == ComplexF64
    rec = nsct_inverse(c)
    @test maximum(abs, rec .- z) < 1.0e-12
    # circular-shift invariance carries over to complex data
    s = (5, 3)
    cs = nsct_forward(circshift(z, s), p)
    for j in 1:p.J, k in eachindex(c.subbands[j])
        @test maximum(abs, cs.subbands[j][k] .- circshift(c.subbands[j][k], s)) < 1.0e-10
    end
end

@testitem "ComplexF32 precision is preserved" tags = [:ct, :nsct] begin
    using Random
    Random.seed!(43)
    p = ContourletParams(J = 2, L_array = [2, 3])
    z = ComplexF32.(complex.(randn(Float32, 64, 64), randn(Float32, 64, 64)))
    c = ct_forward(z, p)
    @test eltype(c.coarse) == ComplexF32
    @test eltype(c.params) == Float32           # filter precision tracks the data
    @test maximum(abs, ct_inverse(c) .- z) < 1.0f-4
    n = nsct_forward(z, p)
    @test eltype(n.coarse) == ComplexF32
    @test maximum(abs, nsct_inverse(n) .- z) < 1.0f-4
end

@testitem "complex workspace path matches allocating" tags = [:ct, :nsct] begin
    using Random
    Random.seed!(44)
    p = ContourletParams(J = 2, L_array = [1, 2])
    z = complex.(randn(64, 64), randn(64, 64))

    ws = make_workspace(ComplexF64, (64, 64), p)
    @test ws isa ContourletWorkspace{ComplexF64, Float64}
    cw = ct_forward(z, p; workspace = ws)
    @test cw.coarse ≈ ct_forward(z, p).coarse
    @test maximum(abs, ct_inverse(cw) .- z) < 1.0e-12

    wsn = make_nsct_workspace(ComplexF64, (64, 64), p)
    nw = nsct_forward(z, p; workspace = wsn)
    @test nw.coarse ≈ nsct_forward(z, p).coarse
end

@testitem "primitive conv2d_sep with complex data, real filter" tags = [:primitives] begin
    using Random
    Random.seed!(45)
    h = Float64[0.25, 0.5, 0.25]
    x = randn(16, 16)
    y = randn(16, 16)
    rz = conv2d_sep(complex.(x, y), h, h)
    @test eltype(rz) == ComplexF64
    @test real(rz) ≈ conv2d_sep(x, h, h)
    @test imag(rz) ≈ conv2d_sep(y, h, h)
end
