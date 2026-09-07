@testitem "ContourletParams3D — construction and validation" tags = [:nsct3d] begin
    p = ContourletParams3D(J = 3, L_array = [1, 2, 2])
    @test p.J == 3
    @test p.L_array == [1, 2, 2]
    @test eltype(p) == Float64
    @test nsct3d_nsubbands(p, 1) == 6
    @test nsct3d_nsubbands(p, 2) == 12
    @test occursin("ContourletParams3D", sprint(show, p))
    @test_throws ArgumentError ContourletParams3D(J = 0, L_array = Int[])
    @test_throws ArgumentError ContourletParams3D(J = 2, L_array = [1])
    @test_throws ArgumentError ContourletParams3D(J = 1, L_array = [-1])
    @test_throws ArgumentError ContourletParams3D(J = 1, L_array = [1], order = 0)
    @test_throws ArgumentError ContourletParams3D(J = 1, L_array = [1], poly = -1)
    @test_throws ArgumentError ContourletParams3D(J = 1, L_array = [1], transition = -1)
end

@testitem "nsct3d — perfect reconstruction" tags = [:nsct3d] begin
    using Random
    Random.seed!(90)
    for dims in ((16, 16, 16), (12, 20, 8))
        x = randn(dims...)
        for (J, L) in ((1, [0]), (2, [1, 2]), (3, [1, 1, 2]))
            p = ContourletParams3D(J = J, L_array = L, order = 6)
            c = nsct3d_forward(x, p)
            @test length(c.subbands) == J
            @test all(j -> length(c.subbands[j]) == 3 * 2^L[j], 1:J)
            @test all(j -> all(b -> size(b) == dims, c.subbands[j]), 1:J)
            @test size(c.coarse) == dims
            @test maximum(abs, nsct3d_inverse(c, p) .- x) < 1.0e-12
        end
    end
end

@testitem "nsct3d — shift invariance" tags = [:nsct3d] begin
    using Random
    Random.seed!(91)
    x = randn(16, 16, 16)
    p = ContourletParams3D(J = 2, L_array = [1, 2], order = 6)
    s = (3, -2, 5)
    a = nsct3d_forward(x, p)
    b = nsct3d_forward(circshift(x, s), p)
    @test maximum(abs, b.coarse .- circshift(a.coarse, s)) < 1.0e-10
    for j in 1:2, k in eachindex(a.subbands[j])
        @test maximum(abs, b.subbands[j][k] .- circshift(a.subbands[j][k], s)) < 1.0e-10
    end
end

@testitem "nsct3d — a constant volume stays in the coarse residual" tags = [:nsct3d] begin
    x = fill(1.75, 16, 16, 16)
    p = ContourletParams3D(J = 2, L_array = [1, 1], order = 6)
    c = nsct3d_forward(x, p)
    @test maximum(abs, c.coarse .- 1.75) < 1.0e-12
    for j in 1:2, b in c.subbands[j]
        @test maximum(abs, b) < 1.0e-12
    end
end

@testitem "nsct3d — streaming forward matches the materialised transform" tags = [:nsct3d] begin
    using Random
    Random.seed!(92)
    x = randn(16, 16, 16)
    p = ContourletParams3D(J = 2, L_array = [1, 2], order = 6)
    ref = nsct3d_forward(x, p)
    seen = Tuple{Int, Int}[]
    coarse = nsct3d_forward(x, p) do j, k, band
        push!(seen, (j, k))
        @test maximum(abs, band .- ref.subbands[j][k]) == 0.0
    end
    @test maximum(abs, coarse .- ref.coarse) == 0.0
    @test seen == [(j, k) for j in 1:2 for k in 1:(3 * 2^p.L_array[j])]
end

@testitem "nsct3d — streaming inverse matches" tags = [:nsct3d] begin
    using Random
    Random.seed!(93)
    x = randn(16, 16, 16)
    p = ContourletParams3D(J = 2, L_array = [1, 2], order = 6)
    c = nsct3d_forward(x, p)
    r = nsct3d_inverse(c.coarse, p) do j, k
        return c.subbands[j][k]
    end
    @test maximum(abs, r .- x) < 1.0e-12
    @test_throws DimensionMismatch nsct3d_inverse(c.coarse, p) do j, k
        return zeros(4, 4, 4)
    end
end

@testitem "nsct3d — complex equals real + i·imag bit-for-bit" tags = [:nsct3d] begin
    using Random
    Random.seed!(94)
    xr = randn(16, 16, 16); xi = randn(16, 16, 16)
    p = ContourletParams3D(J = 2, L_array = [1, 1], order = 6)
    cc = nsct3d_forward(complex.(xr, xi), p)
    cr = nsct3d_forward(xr, p)
    ci = nsct3d_forward(xi, p)
    @test maximum(abs, cc.coarse .- complex.(cr.coarse, ci.coarse)) == 0.0
    for j in 1:2, k in eachindex(cc.subbands[j])
        @test maximum(abs, cc.subbands[j][k] .- complex.(cr.subbands[j][k], ci.subbands[j][k])) == 0.0
    end
    @test maximum(abs, nsct3d_inverse(cc, p) .- complex.(xr, xi)) < 1.0e-12
end

@testitem "nsct3d — Float32" tags = [:nsct3d] begin
    using Random
    Random.seed!(95)
    x = randn(Float32, 16, 16, 16)
    p = ContourletParams3D(J = 2, L_array = [1, 1], order = 6)
    c = nsct3d_forward(x, p)
    @test eltype(c) == Float32
    @test eltype(c.coarse) == Float32
    @test maximum(abs, nsct3d_inverse(c, p) .- x) < 1.0f-4
end

@testitem "nsct3d — workspace reuse is bit-identical and validated" tags = [:nsct3d] begin
    using Random
    Random.seed!(96)
    x = randn(16, 16, 16)
    p = ContourletParams3D(J = 2, L_array = [1, 2], order = 6)
    ws = make_nsct3d_workspace(p, (16, 16, 16), Float64)
    a = nsct3d_forward(x, p)
    b = nsct3d_forward(x, p; workspace = ws)
    c = nsct3d_forward(x, p; workspace = ws)   # second call must not drift
    @test maximum(abs, b.coarse .- a.coarse) == 0.0
    @test maximum(abs, c.coarse .- a.coarse) == 0.0
    for j in 1:2, k in eachindex(a.subbands[j])
        @test maximum(abs, b.subbands[j][k] .- a.subbands[j][k]) == 0.0
        @test maximum(abs, c.subbands[j][k] .- a.subbands[j][k]) == 0.0
    end
    @test maximum(abs, nsct3d_inverse(b, p; workspace = ws) .- x) < 1.0e-12

    @test_throws DimensionMismatch nsct3d_forward(randn(8, 8, 8), p; workspace = ws)
    @test_throws ArgumentError nsct3d_forward(complex.(x, x), p; workspace = ws)
    wsj = make_nsct3d_workspace(ContourletParams3D(J = 1, L_array = [1]), (16, 16, 16), Float64)
    @test_throws ArgumentError nsct3d_forward(x, p; workspace = wsj)
    # Per-scale directional depth and the `tight` normalisation are baked into
    # the banks, so a workspace that disagrees with `params` must be rejected
    # rather than silently overriding it.
    wsl = make_nsct3d_workspace(ContourletParams3D(J = 2, L_array = [2, 2], order = 6), (16, 16, 16), Float64)
    @test_throws ArgumentError nsct3d_forward(x, p; workspace = wsl)
    wst = make_nsct3d_workspace(
        ContourletParams3D(J = 2, L_array = [1, 2], order = 6, tight = false),
        (16, 16, 16), Float64
    )
    @test_throws ArgumentError nsct3d_forward(x, p; workspace = wst)
end

@testitem "nsct3d — show and coefficient container" tags = [:nsct3d] begin
    using Random
    Random.seed!(97)
    x = randn(8, 8, 8)
    p = ContourletParams3D(J = 1, L_array = [1], order = 4)
    c = nsct3d_forward(x, p)
    s = sprint(show, c)
    @test occursin("NSCT3DCoefficients", s)
    @test occursin("J=1", s)
end
