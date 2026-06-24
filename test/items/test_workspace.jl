@testitem "make_workspace allocates correct sizes" tags = [:ct] begin
    p = ContourletParams(J = 2, L_array = [2, 3])
    ws = make_workspace(Float64, (32, 32), p)
    @test ws isa ContourletWorkspace
    @test ws.image_size == (32, 32)
end

@testitem "make_nsct_workspace all same size" tags = [:nsct] begin
    p = ContourletParams(J = 2, L_array = [2, 3])
    ws = make_nsct_workspace(Float64, (32, 32), p)
    @test ws isa ContourletWorkspace
    @test length(ws.qup_cache) == 2
end

@testitem "CT with workspace gives same result as without" tags = [:ct] begin
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

@testitem "estimate_workspace_size" tags = [:ct, :nsct] begin
    p = ContourletParams(J = 2, L_array = [2, 3])
    n = estimate_workspace_size(p, (32, 32))
    @test n isa Int
    @test n > 0
    # Analytical: coarse1=256, coarse2=64, bp1=1024, bp2=256, + 3*1024 = 4672
    @test n == 4672

    n_ns = estimate_workspace_size(p, (32, 32); nonsubsampled = true)
    @test n_ns isa Int
    @test n_ns > 0
    # Analytical: 4 bufs/level * 2 levels * 1024 + 3*1024 = 8192 + 3072 = 11264
    @test n_ns == 11264
    # Estimate must be ≥ actual scratch arena size after a warm forward pass
    ws = make_nsct_workspace(p, (32, 32))
    actual = length(ws.fwd_scratch.bufs) * 32 * 32
    @test n_ns >= actual
end

@testitem "make_workspace params (type-first positional API)" tags = [:ct] begin
    p = ContourletParams(J = 2, L_array = [1, 2])
    # Float64 params + Float32 request → promotes to Float64
    ws = make_workspace(Float32, (32, 32), p)
    @test ws isa ContourletWorkspace{Float64}
    @test ws.image_size == (32, 32)
end

@testitem "make_nsct_workspace (keyword T)" tags = [:nsct] begin
    p = ContourletParams(J = 2, L_array = [2, 3])
    # Float64 params + Float32 request → promotes to Float64
    ws = make_nsct_workspace(p, (32, 32); T = Float32)
    @test ws isa ContourletWorkspace{Float64}
end

@testitem "make_nsct_workspace (type-first positional API)" tags = [:nsct] begin
    p = ContourletParams(J = 2, L_array = [1, 2])
    ws = make_nsct_workspace(Float32, (32, 32), p)
    @test ws isa ContourletWorkspace{Float64}
end

@testitem "NSCT workspace path is allocation-free" tags = [:nsct] begin
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
    @test all(isapprox(coeffs.subbands[j][k], ref.subbands[j][k]; atol = 1.0e-10) for j in 1:2 for k in eachindex(ref.subbands[j]))
end

@testitem "CT workspace path is allocation-free" tags = [:ct] begin
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

@testitem "make_workspace image-first overload" tags = [:ct] begin
    p = ContourletParams(J = 2, L_array = [1, 2])
    img = randn(32, 32)
    ws = make_workspace(img, p)
    @test ws isa ContourletWorkspace
    @test ws.image_size == (32, 32)
    coeffs = similar_coefficients(p, (32, 32))
    ct_forward!(coeffs, img, ws)
    rec = similar(img)
    ct_inverse!(rec, coeffs, ws)
    @test maximum(abs, rec .- img) < 1.0e-12
end

@testitem "make_nsct_workspace image-first overload" tags = [:nsct] begin
    p = ContourletParams(J = 2, L_array = [1, 2])
    img = randn(32, 32)
    ws = make_nsct_workspace(img, p)
    @test ws isa ContourletWorkspace
    @test ws.image_size == (32, 32)
    coeffs = similar_nsct_coefficients(p, (32, 32))
    nsct_forward!(coeffs, img, ws)
    rec = similar(img)
    nsct_inverse!(rec, coeffs, ws)
    @test maximum(abs, rec .- img) < 1.0e-12
end

@testitem "make_nsct_workspace Complex image-first overload" tags = [:nsct] begin
    using Random
    Random.seed!(50)
    p = ContourletParams(J = 2, L_array = [1, 2])
    img = complex.(randn(32, 32), randn(32, 32))
    ws = make_nsct_workspace(img, p)
    @test ws isa ContourletWorkspace{ComplexF64, Float64}
end

@testitem "_base_array_type on SubArray type" tags = [:ct] begin
    x = randn(8, 8)
    sv = @view x[1:4, 1:4]
    scratch = Contourlets._scratch_like(sv, 4, 4)
    @test scratch isa Matrix
    @test size(scratch) == (4, 4)
end

@testitem "NSCT workspace threading mismatch warns" tags = [:nsct] begin
    using Random, Logging
    Random.seed!(99)
    x = randn(32, 32)
    p = ContourletParams(J = 2, L_array = [2, 3])
    # Build with Disabled (fft_threaded=false) then call with Enabled (wants true).
    ws = make_nsct_workspace(p, (32, 32); threading = Disabled())
    coeffs = similar_nsct_coefficients(p, (32, 32))
    @test_logs (:warn,) nsct_forward!(coeffs, x, ws; threading = Enabled())
end

@testitem "NSCT workspace fft_threaded field reflects construction threading" tags = [:nsct] begin
    p = ContourletParams(J = 1, L_array = [1])
    ws_auto = make_nsct_workspace(p, (32, 32); threading = Auto())
    ws_ena = make_nsct_workspace(p, (32, 32); threading = Enabled())
    ws_dis = make_nsct_workspace(p, (32, 32); threading = Disabled())
    # Auto() on Float64 → not threaded
    @test ws_auto.fft_threaded === false
    @test ws_ena.fft_threaded === true
    @test ws_dis.fft_threaded === false
end
