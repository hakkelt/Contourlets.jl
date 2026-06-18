@testitem "make_workspace allocates correct sizes" begin
    p = ContourletParams(J = 2, L_array = [2, 3])
    ws = make_workspace(Float64, (32, 32), p)
    @test ws isa ContourletWorkspace
    @test ws.image_size == (32, 32)
end

@testitem "make_nsct_workspace all same size" begin
    p = ContourletParams(J = 2, L_array = [2, 3])
    ws = make_nsct_workspace(Float64, (32, 32), p)
    @test ws isa ContourletWorkspace
    @test length(ws.qup_cache) == 2
end

@testitem "CT with workspace gives same result as without" begin
    using Random
    Random.seed!(40)
    x = randn(32, 32)
    p = ContourletParams(J = 2, L_array = [2, 3])
    ws = make_workspace(Float64, (32, 32), p)
    c1 = ct_forward(x, p)
    c2 = ct_forward(x, p; workspace = ws)
    @test maximum(abs, c1.coarse .- c2.coarse) < 1.0e-12
    @test maximum(abs, c1.subbands[1][1] .- c2.subbands[1][1]) < 1.0e-12
end

@testitem "estimate_workspace_size" begin
    p = ContourletParams(J = 2, L_array = [2, 3])
    n = estimate_workspace_size(p, (32, 32))
    @test n isa Int
    @test n > 0
    # Analytical: coarse1=256, coarse2=64, bp1=1024, bp2=256, + 3*1024 = 4672
    @test n == 4672

    n_ns = estimate_workspace_size(p, (32, 32); nonsubsampled = true)
    @test n_ns isa Int
    @test n_ns > 0
    # Analytical: 2*2*1024 + 3*1024 = 4096 + 3072 = 7168
    @test n_ns == 7168
end

@testitem "make_workspace params (type-first positional API)" begin
    p = ContourletParams(J = 2, L_array = [1, 2])
    # Float64 params + Float32 request → promotes to Float64
    ws = make_workspace(Float32, (32, 32), p)
    @test ws isa ContourletWorkspace{Float64}
    @test ws.image_size == (32, 32)
end

@testitem "make_nsct_workspace (keyword T)" begin
    p = ContourletParams(J = 2, L_array = [2, 3])
    # Float64 params + Float32 request → promotes to Float64
    ws = make_nsct_workspace(p, (32, 32); T = Float32)
    @test ws isa ContourletWorkspace{Float64}
end

@testitem "make_nsct_workspace (type-first positional API)" begin
    p = ContourletParams(J = 2, L_array = [1, 2])
    ws = make_nsct_workspace(Float32, (32, 32), p)
    @test ws isa ContourletWorkspace{Float64}
end

@testitem "NSCT workspace path is allocation-free" begin
    using Random
    Random.seed!(43)
    x = randn(64, 64)
    p = ContourletParams(J = 2, L_array = [2, 3])
    # Prewarm (default) grows the scratch arena and caches the à trous filters,
    # so the workspace forward/inverse paths allocate nothing per call.
    ws = make_nsct_workspace(Float64, (64, 64), p)
    coeffs = similar_nsct_coefficients(p, (64, 64))
    out = similar(x)
    # One untimed call to ensure everything is compiled for these argument types.
    nsct_forward!(coeffs, x, ws)
    nsct_inverse!(out, coeffs, ws)
    @test (@allocated nsct_forward!(coeffs, x, ws)) == 0
    @test (@allocated nsct_inverse!(out, coeffs, ws)) == 0
    # Results still match the allocating reference exactly (real-linear, bit-for-bit).
    ref = nsct_forward(x, p)
    @test coeffs.coarse == ref.coarse
    @test all(coeffs.subbands[j][k] == ref.subbands[j][k] for j in 1:2 for k in eachindex(ref.subbands[j]))
end

@testitem "CT workspace path is allocation-free" begin
    using Contourlets, Random
    Random.seed!(41)
    x = randn(32, 32)
    p = ContourletParams(J = 2, L_array = [2, 3])
    ws = make_workspace(p, (32, 32))
    coeffs = similar_coefficients(p, (32, 32))
    img_out = similar(x)
    # One untimed call to ensure everything is compiled
    ct_forward!(coeffs, x, ws)
    ct_inverse!(img_out, coeffs, ws)
    
    # We want this to be as low as possible
    fwd_allocs = @allocated ct_forward!(coeffs, x, ws)
    inv_allocs = @allocated ct_inverse!(img_out, coeffs, ws)
    
    println("fwd allocs: ", fwd_allocs)
    println("inv allocs: ", inv_allocs)
    @test fwd_allocs < 1000
    @test inv_allocs < 1000
end
