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
    @test length(ws.nsdfb_banks) == 2
    @test ws.bp_dims == [(32, 32), (32, 32)]
    # rfft half-plane masks for real data
    @test all(size(m) == (17, 32) for b in ws.nsdfb_banks for m in b.ana)
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
    nsct_forward!(coeffs, x, p; workspace = ws)
    nsct_inverse!(out, coeffs, p; workspace = ws)
    # The workspace computation itself is allocation-free; a small constant overhead
    # (~500 bytes) is attributable to Julia's kwarg dispatch machinery in @allocated.
    @test (@allocated nsct_forward!(coeffs, x, p; workspace = ws)) < 1000
    @test (@allocated nsct_inverse!(out, coeffs, p; workspace = ws)) < 1000
    # Results match the allocating reference, which convolves where the workspace
    # multiplies spectra: equal to rounding.
    ref = nsct_forward(x, p)
    @test isapprox(coeffs.coarse, ref.coarse; atol = 1.0e-12)
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
    ct_forward!(coeffs, x, p; workspace = ws)
    ct_inverse!(img_out, coeffs, p; workspace = ws)

    # We want this to be as low as possible
    fwd_allocs = @allocated ct_forward!(coeffs, x, p; workspace = ws)
    inv_allocs = @allocated ct_inverse!(img_out, coeffs, p; workspace = ws)

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
    ct_forward!(coeffs, img, p; workspace = ws)
    rec = similar(img)
    ct_inverse!(rec, coeffs, p; workspace = ws)
    @test maximum(abs, rec .- img) < 1.0e-12
end

@testitem "make_nsct_workspace image-first overload" tags = [:nsct] begin
    p = ContourletParams(J = 2, L_array = [1, 2])
    img = randn(32, 32)
    ws = make_nsct_workspace(img, p)
    @test ws isa ContourletWorkspace
    @test ws.image_size == (32, 32)
    coeffs = similar_nsct_coefficients(p, (32, 32))
    nsct_forward!(coeffs, img, p; workspace = ws)
    rec = similar(img)
    nsct_inverse!(rec, coeffs, p; workspace = ws)
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

@testitem "NSCT/TICT workspace threading is chosen per call" tags = [:nsct, :tict] begin
    using Random
    Random.seed!(99)
    x = randn(32, 32)
    p = ContourletParams(J = 3, L_array = [0, 2, 3])
    ws = make_nsct_workspace(p, (32, 32))
    @test length(ws.fft_tasks) == Threads.nthreads()
    c_ser = nsct_forward!(similar_nsct_coefficients(p, (32, 32)), x, p; workspace = ws, threading = Disabled())
    c_par = nsct_forward!(similar_nsct_coefficients(p, (32, 32)), x, p; workspace = ws, threading = Enabled())
    @test all(c_ser.subbands[j][k] == c_par.subbands[j][k] for j in 1:3 for k in eachindex(c_ser.subbands[j]))
    @test c_ser.coarse == c_par.coarse
    y_ser = nsct_inverse!(similar(x), c_ser, p; workspace = ws, threading = Disabled())
    y_par = nsct_inverse!(similar(x), c_ser, p; workspace = ws, threading = Enabled())
    @test y_par ≈ y_ser atol = 1.0e-13
    @test y_par ≈ x atol = 1.0e-12

    wt = make_tict_workspace(p, (32, 32))
    t_ser = tict_forward!(similar_tict_coefficients(p, (32, 32)), x, p; workspace = wt, threading = Disabled())
    t_par = tict_forward!(similar_tict_coefficients(p, (32, 32)), x, p; workspace = wt, threading = Enabled())
    @test all(t_ser.subbands[i][c][k] == t_par.subbands[i][c][k] for i in 1:3 for c in 1:4 for k in eachindex(t_ser.subbands[i][c]))
    z_par = tict_inverse!(similar(x), t_ser, p; workspace = wt, threading = Enabled())
    @test z_par ≈ tict_inverse!(similar(x), t_ser, p; workspace = wt, threading = Disabled()) atol = 1.0e-13
    @test z_par ≈ x atol = 1.0e-12
end

@testitem "make_workspace return type is concrete (no Union)" tags = [:ct] begin
    # `make_workspace`'s inferred return type must stay a single concrete type across
    # ladder and non-ladder filter pairs, never a `Union` of two `ContourletWorkspace{...}`
    # instantiations — a Union return type is invisible to `@test_call`/`@test_opt` run
    # against `make_workspace` alone, but poisons any downstream package that builds a type
    # parameter from `typeof(make_workspace(...))` (e.g. AbstractOperators.jl's
    # ContourletOperators subpackage does `ContourletOp{...,typeof(workspace),...}`),
    # producing an unresolvable-`convert` JET error at that call site instead.
    p = ContourletParams(J = 2, L_array = [1, 2])
    rt = Base.return_types(make_workspace, (typeof(p), Tuple{Int, Int}))
    @test length(rt) == 1
    @test isconcretetype(only(rt))
end

@testitem "make_workspace with non-ladder (modulation-mode) filter pair" tags = [:ct] begin
    using Random
    Random.seed!(44)
    # Modulation-mode (non-ladder) filter pair: the 4-arg QuincunxFilterPair constructor
    # used throughout the test suite (e.g. test_filters.jl, test_dfb.jl) for the Haar pair.
    haar = QuincunxFilterPair([0.5 0.5], [1.0 1.0], (1, 1), (1, 2))
    @test !Contourlets.is_ladder(haar)
    p = ContourletParams(J = 1, L_array = [1], dfb_filters = haar)

    ws = make_workspace(p, (16, 16))
    # No ladder, no DFB runners — but the same workspace type as a ladder pair gets,
    # so `make_workspace`'s return type stays stable across filter modes.
    @test isempty(ws.dfb_runners)
    @test typeof(ws) == typeof(make_workspace(ContourletParams(J = 1, L_array = [1]), (16, 16)))

    x = randn(16, 16)
    coeffs = similar_coefficients(p, (16, 16))
    rec = similar(x)
    ct_forward!(coeffs, x, p; workspace = ws)
    ct_inverse!(rec, coeffs, p; workspace = ws)
    @test maximum(abs, rec .- x) < 1.0e-10

    ref = ct_forward(x, p)
    @test maximum(abs, coeffs.coarse .- ref.coarse) < 1.0e-10
end

@testitem "_with_filter_cache sets and restores the task-local cache" tags = [:ct] begin
    @test Contourlets._active_filter_cache() === nothing
    cache = Dict(:a => 1)
    inner = Contourlets._with_filter_cache(cache) do
        @test Contourlets._active_filter_cache() === cache
        # Nesting restores the outer cache afterwards.
        Contourlets._with_filter_cache(:inner) do
            @test Contourlets._active_filter_cache() === :inner
        end
        @test Contourlets._active_filter_cache() === cache
        42
    end
    @test inner == 42
    @test Contourlets._active_filter_cache() === nothing
    # Restored even when the body throws.
    @test_throws ErrorException Contourlets._with_filter_cache(() -> error("boom"), cache)
    @test Contourlets._active_filter_cache() === nothing
end

@testitem "NSCT workspace with an unsplit level (L = 0)" tags = [:nsct] begin
    using Random
    Random.seed!(90)
    x = randn(32, 32)
    p = ContourletParams(J = 2, L_array = [2, 0])
    ws = make_nsct_workspace(p, size(x))
    @test ws.nsdfb_banks[2] === nothing
    c = similar_nsct_coefficients(p, size(x))
    nsct_forward!(c, x, p; workspace = ws)
    ref = nsct_forward(x, p)
    @test length(c.subbands[2]) == 1
    @test c.subbands[2][1] ≈ ref.subbands[2][1] atol = 1.0e-12
    out = similar(x)
    nsct_inverse!(out, c, p; workspace = ws)
    @test maximum(abs, out .- x) < 1.0e-12
end

@testitem "DFB falls back to its own bank when the workspace has none" tags = [:ct] begin
    using Random
    Random.seed!(91)
    p = ContourletParams(J = 1, L_array = [2])
    ws = make_workspace(p, (32, 32))
    bp = randn(32, 32)
    # The workspace holds a depth-2 bank for 32×32 only: depth 3 and another size
    # must build their own and give the same result as no workspace at all.
    @test Contourlets._workspace_runner(ws, (32, 32), 3, :both) === nothing
    @test Contourlets._workspace_runner(ws, (16, 16), 2, :both) === nothing
    a = dfb_decompose(bp, 3, Q2345; workspace = ws)
    b = dfb_decompose(bp, 3, Q2345)
    @test all(a .== b)
    @test maximum(abs, dfb_reconstruct(a, Q2345; workspace = ws) .- bp) < 1.0e-12
end

@testitem "ScratchArena resizes a slot whose requested size changes" tags = [:ct] begin
    a = Contourlets.ScratchArena{Matrix{Float64}}()
    ref = zeros(1, 1)
    b1 = Contourlets._arena_take!(a, ref, 4, 4)
    Contourlets._arena_reset!(a)
    @test Contourlets._arena_take!(a, ref, 4, 4) === b1      # same slot, same size: reused
    Contourlets._arena_reset!(a)
    b2 = Contourlets._arena_take!(a, ref, 2, 3)              # same slot, new size: replaced
    @test size(b2) == (2, 3)
    @test length(a.bufs) == 1
end

@testitem "_allocate_zeros generic (non-Array) matrix path" tags = [:ct] begin
    # UnionAll Array type → `zeros` fast path.
    A = Contourlets._allocate_zeros(Matrix, Float32, (3, 4))
    @test A isa Matrix{Float32}
    @test iszero(A)
    # A non-Array concrete matrix type goes through `similar` + `fill!`
    # (device arrays use this path; see test_gpu.jl for JLArrays).
    B = Contourlets._allocate_zeros(BitMatrix, Bool, (3, 4))
    @test B isa BitMatrix
    @test size(B) == (3, 4)
    @test !any(B)
end
