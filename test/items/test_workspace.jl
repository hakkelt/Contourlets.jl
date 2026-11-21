@testitem "make_workspace allocates correct sizes" tags = [:ct] begin
    t = CT(scales = 2, direction_levels = [2, 3])
    ws = make_workspace(t, Float64, (32, 32))
    @test ws isa CTWorkspace
    @test size(ws) == (32, 32)
    @test ws.bp_dims == [(32, 32), (16, 16)]
    @test [r.bank.levels for r in ws.dfb_runners] == [2, 3]
end

@testitem "NSCT workspace: all masks on the rfft grid" tags = [:nsct] begin
    t = NSCT(scales = 2, direction_levels = [2, 3])
    ws = make_workspace(t, Float64, (32, 32))
    @test ws isa NSCTWorkspace
    e = ws.engine
    @test length(e.banks) == 2
    @test length(e.items) == 4 + 8
    # rfft half-plane masks for real data
    @test all(size(m) == (17, 32) for b in e.banks for m in b.ana)
    @test all(size(m) == (17, 32) for m in e.pyramid_ana)
end

@testitem "workspaces run at the data's precision" tags = [:ct, :nsct, :tict] begin
    for t in (CT(scales = 2, direction_levels = [1, 2]), NSCT(scales = 2, direction_levels = [1, 2]), TICT(scales = 2, direction_levels = [1, 2]), STICT(scales = 2, direction_levels = [1, 2]))
        # Float64 filters, Float32 data: a Float32 workspace with converted filters.
        ws = make_workspace(t, Float32, (32, 32))
        @test eltype(ws) == Float32
        @test eltype(ws.params.lp_filters) == Float32
        @test transform(ws) === t
        wsc = make_workspace(t, randn(ComplexF32, 32, 32))
        @test eltype(wsc) == ComplexF32
        @test eltype(wsc.params.lp_filters) == Float32
    end
end

@testitem "NSCT workspace path is allocation-free" tags = [:nsct] begin
    using Random
    Random.seed!(43)
    x = randn(64, 64)
    t = NSCT(scales = 2, direction_levels = [2, 3])
    # Prewarm (default) runs one transform pair, so the workspace paths allocate
    # nothing per call.
    ws = make_workspace(t, Float64, (64, 64))
    coeffs = similar_coefficients(t, x)
    out = similar(x)
    let t = t, ws = ws, x = x, coeffs = coeffs, out = out
        allocs(f) = (f(); @allocated f())
        @test allocs(() -> analyze!(coeffs, t, x; workspace = ws, threaded = false)) == 0
        @test allocs(() -> synthesize!(out, t, coeffs; workspace = ws, threaded = false)) == 0
        @test allocs(() -> analyze!(coeffs, ws, x; threaded = false)) == 0
        @test allocs(() -> synthesize!(out, ws, coeffs; threaded = false)) == 0
    end
    # Results match the allocating reference, which convolves where the workspace
    # multiplies spectra: equal to rounding.
    ref = analyze(t, x)
    @test isapprox(subband(t, coeffs, 1), subband(t, ref, 1); atol = 1.0e-12)
    @test isapprox(coeffs, ref; atol = 1.0e-10)
end

@testitem "CT workspace path is allocation-free" tags = [:ct] begin
    using Random
    Random.seed!(41)
    x = randn(32, 32)
    t = CT(scales = 2, direction_levels = [2, 3])
    ws = make_workspace(t, x)
    coeffs = similar_coefficients(t, x)
    img_out = similar(x)
    let t = t, ws = ws, x = x, coeffs = coeffs, img_out = img_out
        allocs(f) = (f(); @allocated f())
        @test allocs(() -> analyze!(coeffs, t, x; workspace = ws)) == 0
        @test allocs(() -> synthesize!(img_out, t, coeffs; workspace = ws)) == 0
        @test allocs(() -> analyze!(coeffs, ws, x)) == 0
        @test allocs(() -> synthesize!(img_out, ws, coeffs)) == 0
    end
end

@testitem "make_workspace from an image" tags = [:ct, :nsct] begin
    img = randn(32, 32)
    for t in (CT(scales = 2, direction_levels = [1, 2]), NSCT(scales = 2, direction_levels = [1, 2]))
        ws = make_workspace(t, img)
        @test size(ws) == (32, 32)
        @test eltype(ws) == Float64
        coeffs = similar_coefficients(t, img)
        analyze!(coeffs, t, img; workspace = ws)
        rec = similar(img)
        synthesize!(rec, t, coeffs; workspace = ws)
        @test maximum(abs, rec .- img) < 1.0e-12
    end
end

@testitem "NSCT workspace from a complex image" tags = [:nsct] begin
    using Random
    Random.seed!(50)
    img = complex.(randn(32, 32), randn(32, 32))
    ws = make_workspace(NSCT(scales = 2, direction_levels = [1, 2]), img)
    @test ws isa NSCTWorkspace{ComplexF64, Float64}
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
    for t in (NSCT(scales = 3, direction_levels = [0, 2, 3]), TICT(scales = 3, direction_levels = [0, 2, 3]))
        ws = make_workspace(t, Float64, (32, 32))
        @test length(ws.engine.buffers) == min(Threads.nthreads(), length(ws.engine.items))
        c_ser = analyze!(similar_coefficients(t, x), t, x; workspace = ws, threaded = false)
        c_par = analyze!(similar_coefficients(t, x), t, x; workspace = ws, threaded = true)
        @test c_ser == c_par
        y_ser = synthesize!(similar(x), t, c_ser; workspace = ws, threaded = false)
        y_par = synthesize!(similar(x), t, c_ser; workspace = ws, threaded = true)
        @test y_par ≈ y_ser atol = 1.0e-13
        @test y_par ≈ x atol = 1.0e-12
    end
end

@testitem "make_workspace return type is concrete (no Union)" tags = [:ct] begin
    # `make_workspace`'s inferred return type must stay a single concrete type: a
    # downstream package that builds a type parameter from
    # `typeof(make_workspace(...))` cannot resolve a `Union`.  Checked for host
    # arrays; a device NSCT/TICT workspace's engine type depends on the runtime
    # `tune` keyword.
    for t in (CT(scales = 2, direction_levels = [1, 2]), STICT(scales = 2, direction_levels = [1, 2]), NSCT(scales = 2, direction_levels = [1, 2]), TICT(scales = 2, direction_levels = [1, 2]))
        for args in ((typeof(t), Type{Float64}, Tuple{Int, Int}), (typeof(t), Matrix{Float32}), (typeof(t), Matrix{ComplexF32}))
            rt = Base.return_types(make_workspace, args)
            @test length(rt) == 1
            @test isconcretetype(only(rt))
        end
        @inferred make_workspace(t, ComplexF64, (16, 16); prewarm = false)
        @inferred make_workspace(t, zeros(Float32, 16, 16); prewarm = false)
    end
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
    t = NSCT(scales = 2, direction_levels = [2, 0])
    ws = make_workspace(t, x)
    # Depth 0 is the identity bank: one channel with all-ones masks.
    b = ws.engine.banks[2]
    @test length(b) == 1
    @test all(isone, b.ana[1]) && all(isone, b.syn[1])
    c = similar_coefficients(t, x)
    analyze!(c, t, x; workspace = ws)
    ref = analyze(t, x)
    @test size(c, 3) == 1 + 4 + 1
    @test subband(t, c, 6) ≈ subband(t, ref, 6) atol = 1.0e-12
    out = similar(x)
    synthesize!(out, t, c; workspace = ws)
    @test maximum(abs, out .- x) < 1.0e-12
end

@testitem "DFB workspace lookup: the held bank or an error" tags = [:ct] begin
    using Random
    Random.seed!(91)
    ws = make_workspace(CT(scales = 1, direction_levels = [2]), Float64, (32, 32))
    bp = randn(32, 32)
    # The workspace holds a depth-2 bank for 32×32 only.
    @test Contourlets._workspace_runner(ws, (32, 32), 2, :both, Q2345, Float64) !== nothing
    @test Contourlets._workspace_runner(nothing, (32, 32), 2, :both, Q2345, Float64) === nothing
    # Equal filters held in another object are accepted.
    @test Contourlets._workspace_runner(ws, (32, 32), 2, :both, QuincunxFilterPair(copy(Q2345.f_ladder)), Float64) !== nothing
    @test_throws "different DFB filters" Contourlets._workspace_runner(ws, (32, 32), 2, :both, QuincunxFilterPair(2 .* Q2345.f_ladder), Float64)
    @test_throws "no DFB for a (32, 32) bandpass at depth 3" Contourlets._workspace_runner(ws, (32, 32), 3, :both, Q2345, Float64)
    @test_throws "no DFB for a (16, 16) bandpass at depth 2" Contourlets._workspace_runner(ws, (16, 16), 2, :both, Q2345, Float64)
    @test_throws "filter precision Float64" Contourlets._workspace_runner(ws, (32, 32), 2, :both, Q2345, Float32)
    a = Contourlets.dfb_decompose(bp, 2, Q2345; workspace = ws)
    @test all(a .== Contourlets.dfb_decompose(bp, 2, Q2345))
    @test maximum(abs, Contourlets.dfb_reconstruct(a, Q2345; workspace = ws) .- bp) < 1.0e-12
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

@testitem "tune is a no-op for host workspaces" tags = [:nsct, :tict] begin
    using Random
    Random.seed!(93)
    x = randn(32, 32)
    for t in (CT(scales = 2, direction_levels = [2, 3]), NSCT(scales = 2, direction_levels = [2, 3]), TICT(scales = 2, direction_levels = [2, 3]), STICT(scales = 2, direction_levels = [2, 3]))
        ws0, ws1 = make_workspace(t, Float64, (32, 32)), make_workspace(t, Float64, (32, 32); tune = true)
        t isa Union{NSCT, TICT} && @test ws1.engine.tuning === nothing
        @test typeof(ws1) === typeof(ws0)
        c0 = analyze(t, x; workspace = ws0)
        c1 = analyze(t, x; workspace = ws1)
        @test c1 == c0
        @test synthesize(t, c1; workspace = ws1) == synthesize(t, c0; workspace = ws0)
    end
end

@testitem "prewarm = false builds the same workspace" tags = [:ct, :nsct, :tict] begin
    using Random
    Random.seed!(94)
    x = randn(32, 32)
    for t in (CT(scales = 2, direction_levels = [2, 3]), NSCT(scales = 2, direction_levels = [2, 3]), TICT(scales = 2, direction_levels = [2, 3]), STICT(scales = 2, direction_levels = [2, 3]))
        ws = make_workspace(t, x; prewarm = false)
        @test analyze(t, x; workspace = ws) == analyze(t, x; workspace = make_workspace(t, x))
    end
end

@testitem "batched forward FFT plan of the device synthesis" tags = [:nsct] begin
    using FFTW
    R = randn(8, 6, 3)
    B = zeros(ComplexF64, 5, 6, 3)
    @test Contourlets._ns_forward_batch_plan(R, B) * R ≈ rfft(R, 1:2)
    Rc = randn(ComplexF64, 8, 6, 3)
    @test Contourlets._ns_forward_batch_plan(Rc, similar(Rc)) * Rc ≈ fft(Rc, 1:2)
end
