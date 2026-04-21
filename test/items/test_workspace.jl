@testitem "make_workspace allocates correct sizes" begin
    using Contourlets
    p = ContourletParams(J = 2, L_array = [2, 3])
    ws = make_workspace(Float64, (32, 32), p)
    @test ws isa ContourletWorkspace
    @test length(ws.coarse_bufs) == 2
    @test size(ws.coarse_bufs[1]) == (16, 16)
    @test size(ws.coarse_bufs[2]) == (8, 8)
end

@testitem "make_nsct_workspace all same size" begin
    using Contourlets
    p = ContourletParams(J = 2, L_array = [2, 3])
    ws = make_nsct_workspace(Float64, (32, 32), p)
    @test ws isa ContourletWorkspace
    for buf in ws.coarse_bufs
        @test size(buf) == (32, 32)
    end
end

@testitem "CT with workspace gives same result as without" begin
    using Contourlets, Random
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
    using Contourlets
    p = ContourletParams(J = 1, L_array = [2])
    ws = make_workspace(Float64, (16, 16), p)
    ws.coarse_bufs[1] .= 999.0
    workspace_clear!(ws)
    @test all(iszero, ws.coarse_bufs[1])
end
