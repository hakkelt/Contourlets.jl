@testitem "make_workspace allocates correct sizes" begin
    p = ContourletParams(J = 2, L_array = [2, 3])
    ws = make_workspace(Float64, (32, 32), p)
    @test ws isa ContourletWorkspace
    @test length(ws.coarse_bufs) == 2
    @test size(ws.coarse_bufs[1]) == (16, 16)
    @test size(ws.coarse_bufs[2]) == (8, 8)
end

@testitem "make_nsct_workspace all same size" begin
    p = ContourletParams(J = 2, L_array = [2, 3])
    ws = make_nsct_workspace(Float64, (32, 32), p)
    @test ws isa ContourletWorkspace
    for buf in ws.coarse_bufs
        @test size(buf) == (32, 32)
    end
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

@testitem "workspace_clear! resets all buffers to zero" begin
    p = ContourletParams(J = 1, L_array = [2])
    ws = make_workspace(Float64, (16, 16), p)
    ws.coarse_bufs[1] .= 999.0
    workspace_clear!(ws)
    @test all(iszero, ws.coarse_bufs[1])
end

@testitem "estimate_workspace_size" begin
    p = ContourletParams(J = 2, L_array = [2, 3])
    n = estimate_workspace_size(p, (32, 32))
    @test n isa Int
    @test n > 0
    # coarse1=256, coarse2=64, bp1=1024, bp2=256, tmp+tmp2+current=3*1024 = 4672
    ws = make_workspace(p, (32, 32))
    manual = sum(length(b) for b in ws.coarse_bufs) +
        sum(length(b) for b in ws.bp_bufs) +
        length(ws.tmp_buf) + length(ws.tmp_buf2) + length(ws.current)
    @test n == manual
    # NSCT variant: all per-level buffers full-size
    nws = make_nsct_workspace(p, (32, 32))
    n_ns = estimate_workspace_size(p, (32, 32); nonsubsampled = true)
    manual_ns = sum(length(b) for b in nws.coarse_bufs) +
        sum(length(b) for b in nws.bp_bufs) +
        length(nws.tmp_buf) + length(nws.tmp_buf2) + length(nws.current)
    @test n_ns == manual_ns
end

@testitem "make_workspace params (type-first positional API)" begin
    p = ContourletParams(J = 2, L_array = [1, 2])
    # Float64 params + Float32 request → promotes to Float64
    ws = make_workspace(Float32, (32, 32), p)
    @test ws isa ContourletWorkspace{Float64}
    @test size(ws.tmp_buf) == (32, 32)
end

@testitem "make_nsct_workspace (keyword T)" begin
    p = ContourletParams(J = 2, L_array = [2, 3])
    # Float64 params + Float32 request → promotes to Float64
    ws = make_nsct_workspace(p, (32, 32); T = Float32)
    @test ws isa ContourletWorkspace{Float64}
    for buf in ws.coarse_bufs
        @test size(buf) == (32, 32)
    end
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

@testitem "CT workspace PR after workspace_clear!" begin
    using Random
    Random.seed!(41)
    x = randn(32, 32)
    p = ContourletParams(J = 2, L_array = [2, 3])
    ws = make_workspace(p, (32, 32))
    coeffs = similar_coefficients(p, (32, 32))
    # First pass
    ct_forward!(coeffs, x, ws)
    rec1 = copy(coeffs.coarse)
    # Clear and re-run
    workspace_clear!(ws)
    ct_forward!(coeffs, x, ws)
    @test maximum(abs, coeffs.coarse .- rec1) < 1.0e-14
end
