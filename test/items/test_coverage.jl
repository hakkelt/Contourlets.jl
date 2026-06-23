# Tests targeting code paths not covered by existing tests.
# Generated to improve coverage from ~85% toward 95%+.

# ── types.jl ──────────────────────────────────────────────────────────────────

@testitem "parabolic_levels J=0 returns empty" tags = [:ct] begin
    ls = parabolic_levels(0)
    @test ls == Int[]
    ls1 = parabolic_levels(0, 2)
    @test ls1 == Int[]
end

@testitem "parabolic_levels large l_j0" tags = [:ct] begin
    ls = parabolic_levels(5, 2)
    @test length(ls) == 5
    @test ls[1] >= ls[end]   # finer scale ≥ coarser scale in direction count
end

# ── workspace.jl ─────────────────────────────────────────────────────────────

@testitem "make_workspace image-first overload" tags = [:ct] begin
    p = ContourletParams(J = 2, L_array = [1, 2])
    img = randn(32, 32)
    ws = make_workspace(img, p)
    @test ws isa ContourletWorkspace
    @test ws.image_size == (32, 32)
    # Verify it works end-to-end
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

# ── conv2d.jl ─────────────────────────────────────────────────────────────────

@testitem "conv2d_sep! periodic and zero boundaries" tags = [:primitives] begin
    using Random
    Random.seed!(60)
    x = randn(8, 8)
    h = Float64[0.25, 0.5, 0.25]
    dst_per = similar(x)
    conv2d_sep!(dst_per, x, h, h; boundary = :periodic)
    # Interior pixels should match symmetric for an 8x8 image (away from border)
    dst_sym = similar(x)
    conv2d_sep!(dst_sym, x, h, h; boundary = :symmetric)
    @test maximum(abs, dst_per[3:6, 3:6] .- dst_sym[3:6, 3:6]) < 1.0e-12

    dst_zero = similar(x)
    conv2d_sep!(dst_zero, x, h, h; boundary = :zero)
    @test size(dst_zero) == size(x)
    @test all(isfinite, dst_zero)
    # Interior should also match
    @test maximum(abs, dst_zero[3:6, 3:6] .- dst_sym[3:6, 3:6]) < 1.0e-12
end

@testitem "conv2d_sep allocating matches in-place" tags = [:primitives] begin
    using Random
    Random.seed!(61)
    x = randn(16, 16)
    h = Float64[0.25, 0.5, 0.25]
    dst_alloc = conv2d_sep(x, h, h; boundary = :periodic)
    dst_ip = similar(x)
    conv2d_sep!(dst_ip, x, h, h; boundary = :periodic)
    @test dst_alloc == dst_ip
end

# ── dfb.jl ────────────────────────────────────────────────────────────────────

@testitem "dfb_decompose! basic" tags = [:directional] begin
    using Random
    Random.seed!(70)
    x = randn(32, 32)
    l = 2
    subbands = [zeros(16, 16) for _ in 1:4]
    dfb_decompose!(subbands, x, l, Q2345)
    ref = dfb_decompose(x, l, Q2345)
    for k in 1:4
        @test maximum(abs, subbands[k] .- ref[k]) < 1.0e-14
    end
end

@testitem "dfb_decompose! with workspace" tags = [:directional] begin
    using Random
    Random.seed!(71)
    x = randn(32, 32)
    l = 2
    p = ContourletParams(J = 1, L_array = [l])
    ws = make_workspace(p, (32, 32))
    subbands = [zeros(16, 16) for _ in 1:4]
    dfb_decompose!(subbands, x, l, Q2345; workspace = ws)
    ref = dfb_decompose(x, l, Q2345)
    for k in 1:4
        @test maximum(abs, subbands[k] .- ref[k]) < 1.0e-14
    end
end

@testitem "dfb_reconstruct! basic" tags = [:directional] begin
    using Random
    Random.seed!(72)
    x = randn(32, 32)
    l = 2
    sbs = dfb_decompose(x, l, Q2345)
    bp = zeros(32, 32)
    dfb_reconstruct!(bp, sbs, Q2345)
    @test maximum(abs, bp .- x) < 1.0e-12
end

@testitem "dfb_reconstruct! with workspace" tags = [:directional] begin
    using Random
    Random.seed!(73)
    x = randn(32, 32)
    l = 2
    p = ContourletParams(J = 1, L_array = [l])
    ws = make_workspace(p, (32, 32))
    sbs = dfb_decompose(x, l, Q2345)
    bp = zeros(32, 32)
    dfb_reconstruct!(bp, sbs, Q2345; workspace = ws)
    @test maximum(abs, bp .- x) < 1.0e-12
end

@testitem "dfb_decompose! invalid subbands length" tags = [:directional] begin
    x = randn(32, 32)
    subbands = [zeros(16, 16)]  # wrong: 1 instead of 4 for l=2
    @test_throws ArgumentError dfb_decompose!(subbands, x, 2, Q2345)
end

@testitem "dfb_reconstruct error on empty subbands" tags = [:directional] begin
    @test_throws ArgumentError dfb_reconstruct(Matrix{Float64}[], Q2345)
end

@testitem "dfb_reconstruct error on non-power-of-2" tags = [:directional] begin
    sbs = [randn(32, 11) for _ in 1:3]   # 3 is not a power of 2
    @test_throws ArgumentError dfb_reconstruct(sbs, Q2345)
end

@testitem "dfb_subband_sizes modulation-mode (ladder=false)" tags = [:directional] begin
    # Modulation-mode DFB: sub_n1 = n1 >> (l÷2), sub_n2 = n2 >> cld(l,2).
    szs = dfb_subband_sizes(64, 64, 2; ladder = false)
    @test length(szs) == 4
    @test all(==((32, 32)), szs)   # n1>>1=32, n2>>1=32

    szs1 = dfb_subband_sizes(64, 64, 1; ladder = false)
    @test length(szs1) == 2
    @test all(s -> s == (64, 32), szs1)   # n1>>0=64, n2>>1=32

    szs0 = dfb_subband_sizes(64, 64, 0; ladder = false)
    @test length(szs0) == 1
    @test szs0[1] == (64, 64)
end

@testitem "DFB threading=Enabled with modulation-mode Haar pair" tags = [:directional] begin
    # This exercises the threaded _dfb_split/_dfb_merge code paths.
    using Random
    Random.seed!(74)
    x = randn(32, 32)
    haar = QuincunxFilterPair([0.5 0.5], [1.0 1.0], (1, 1), (1, 2))
    sbs = dfb_decompose(x, 3, haar; threading = Enabled())
    @test length(sbs) == 8
    rec = dfb_reconstruct(sbs, haar; threading = Enabled())
    @test maximum(abs, rec .- x) < 1.0e-12
end

# ── nsdfb.jl ─────────────────────────────────────────────────────────────────

@testitem "nsdfb_decompose l_levels=0 returns copy" tags = [:directional] begin
    x = randn(16, 16)
    sbs = nsdfb_decompose(x, 0, Q2345, 1)
    @test length(sbs) == 1
    @test sbs[1] == x
    @test sbs[1] !== x   # must be a copy
end

@testitem "nsdfb_reconstruct n=1 returns copy" tags = [:directional] begin
    x = randn(16, 16)
    sbs = [x]
    rec = nsdfb_reconstruct(sbs, Q2345, 1)
    @test rec == x
    @test rec !== x   # must be a copy
end

@testitem "nsdfb_reconstruct non-power-of-2 throws" tags = [:directional] begin
    sbs = [randn(16, 16) for _ in 1:3]
    @test_throws ArgumentError nsdfb_reconstruct(sbs, Q2345, 1)
end

@testitem "_sparse_to_dense round-trips SparseFilter2D" tags = [:directional] begin
    # Build the NSDFB filter bundle and verify round-trip via _sparse_to_dense.
    F = Contourlets._nsdfb_filters(Q2345, Float64)
    # k1 is a SparseFilter2D; round-trip through dense conversion.
    sf = F.k1
    dense = Contourlets._sparse_to_dense(sf)
    @test size(dense) == size(sf)
    # Re-sparsify and compare vals/positions.
    sf2 = Contourlets.SparseFilter2D(dense)
    @test sort(sf2.vals) ≈ sort(sf.vals)
    @test length(sf2.vals) == length(sf.vals)
end

@testitem "_efilter2! dense AbstractMatrix fallback" tags = [:directional] begin
    # The dense _efilter2! fallback is invoked when a non-SparseFilter2D matrix is
    # passed (e.g. from the GPU extension or a custom filter matrix).
    using Random
    Random.seed!(80)
    x = randn(8, 8)
    F = Contourlets._nsdfb_filters(Q2345, Float64)
    # Build a dense matrix from the sparse k1 filter.
    k1_dense = Contourlets._sparse_to_dense(F.k1)
    out_dense = similar(x)
    Contourlets._efilter2!(out_dense, x, k1_dense)
    # Compare against the sparse path.
    out_sparse = similar(x)
    Contourlets._efilter2!(out_sparse, x, F.k1)
    @test maximum(abs, out_dense .- out_sparse) < 1.0e-12
end

@testitem "_zconv2! dense AbstractMatrix fallback" tags = [:directional] begin
    using Random
    Random.seed!(81)
    x = randn(8, 8)
    F = Contourlets._nsdfb_filters(Q2345, Float64)
    M = Contourlets._Q1
    # Build a dense matrix for f1[1].
    f1_dense = Contourlets._sparse_to_dense(F.f1[1])
    out_dense = similar(x)
    Contourlets._zconv2!(out_dense, x, f1_dense, M)
    # Compare against the sparse path.
    out_sparse = similar(x)
    Contourlets._zconv2!(out_sparse, x, F.f1[1], M)
    @test maximum(abs, out_dense .- out_sparse) < 1.0e-12
end

# ── threading.jl ──────────────────────────────────────────────────────────────

@testitem "Enabled threading policy" tags = [:ct] begin
    # Verify that Enabled() forces threading=true for Real data.
    @test Contourlets._use_threading(Enabled(), Float64) === true
    @test Contourlets._use_threading(Enabled(), ComplexF64) === true
end

@testitem "Disabled threading policy" tags = [:ct] begin
    @test Contourlets._use_threading(Disabled(), Float64) === false
    @test Contourlets._use_threading(Disabled(), ComplexF64) === false
end

@testitem "Auto threading policy dispatch" tags = [:ct] begin
    @test Contourlets._use_threading(Auto(), Float64) === false
    @test Contourlets._use_threading(Auto(), ComplexF64) === true
    @test Contourlets._use_threading(Auto(), Float32) === false
    @test Contourlets._use_threading(Auto(), ComplexF32) === true
end

@testitem "CT with threading=Enabled() and Real data" tags = [:ct] begin
    using Random
    Random.seed!(90)
    x = randn(32, 32)
    p = ContourletParams(J = 2, L_array = [2, 3])
    c = ct_forward(x, p; threading = Enabled())
    rec = ct_inverse(c, p; threading = Enabled())
    @test maximum(abs, rec .- x) < 1.0e-12
end

@testitem "NSCT with threading=Enabled() and Real data" tags = [:nsct] begin
    using Random
    Random.seed!(91)
    x = randn(32, 32)
    p = ContourletParams(J = 2, L_array = [2, 3])
    nc = nsct_forward(x, p; threading = Enabled())
    rec = nsct_inverse(nc, p; threading = Enabled())
    @test maximum(abs, rec .- x) < 1.0e-12
end

# ── sefilter2 symmetric path ──────────────────────────────────────────────────
# The _sefilter2_kernel! `sym=true` path is only reached when the modulated
# ladder filter is symmetric (even-function); with Q2345 it is anti-symmetric.
# A short custom symmetric ladder filter triggers both the sym and isodd(L) branches.

@testitem "_sefilter2 symmetric filter path" tags = [:directional] begin
    # f_ladder = [1.0, 2.0, 1.0] → _ladder_modulate → [-1.0, 2.0, -1.0] (symmetric, L=3 odd).
    # Construct a minimal QuincunxFilterPair with this 3-tap symmetric ladder.
    # We only need the DFB decompose/reconstruct to exercise _sefilter2_kernel! sym=true.
    f_sym = [1.0, 2.0, 1.0]
    # Equivalent h_q/g_q placeholders (not used by the ladder path in sefilter2, but
    # required for the struct; use 1-tap identity filters so the sefilter2 call runs).
    qfp_sym = QuincunxFilterPair{Float64}(
        reshape([1.0], 1, 1), reshape([1.0], 1, 1),
        (1, 1), (1, 1), f_sym
    )
    x = randn(32, 32)
    sbs = dfb_decompose(x, 2, qfp_sym)
    @test length(sbs) == 4
    rec = dfb_reconstruct(sbs, qfp_sym)
    @test size(rec) == size(x)
    # PR is structural for any ladder filter, so check approximate reconstruction.
    @test maximum(abs, rec .- x) < 1.0e-10
end

# ── resamp / qpdec / ppdec / pprec error paths ────────────────────────────────

@testitem "_resamp invalid type throws" tags = [:directional] begin
    x = randn(8, 8)
    @test_throws ArgumentError Contourlets._resamp(x, 5)
end

@testitem "_qpdec unsupported type throws" tags = [:directional] begin
    x = randn(8, 8)
    @test_throws ArgumentError Contourlets._qpdec(x, :bad)
end

@testitem "_qprec unsupported type throws" tags = [:directional] begin
    p0 = randn(4, 8); p1 = randn(4, 8)
    @test_throws ArgumentError Contourlets._qprec(p0, p1, :bad)
end

@testitem "_ppdec invalid type throws" tags = [:directional] begin
    x = randn(8, 8)
    @test_throws ArgumentError Contourlets._ppdec(x, 5)
end

@testitem "_pprec invalid type throws" tags = [:directional] begin
    p0 = randn(4, 8); p1 = randn(4, 8)
    @test_throws ArgumentError Contourlets._pprec(p0, p1, 5)
end

# ── _resampz error path ───────────────────────────────────────────────────────

@testitem "_resampz invalid type throws" tags = [:directional] begin
    x = randn(8, 8)
    @test_throws ArgumentError Contourlets._resampz(x, 5)
end

# ── workspace scratch arena SubArray dispatch ─────────────────────────────────

@testitem "_base_array_type on SubArray type" tags = [:ct] begin
    # _base_array_type is called via _scratch_like for SubArray refs;
    # the CPU path uses host Matrix so this exercises the SubArray branch.
    x = randn(8, 8)
    sv = @view x[1:4, 1:4]
    # _scratch_like with a SubArray ref should return a Matrix (the parent type).
    scratch = Contourlets._scratch_like(sv, 4, 4)
    @test scratch isa Matrix
    @test size(scratch) == (4, 4)
end

# ── NSP decompose/reconstruct invalid level ────────────────────────────────────

# ── LP decompose! odd-dimension guard ─────────────────────────────────────────

@testitem "lp_decompose! odd dimension throws" tags = [:pyramid] begin
    x = randn(7, 8)
    c = zeros(4, 4); bp = zeros(7, 8)
    @test_throws ArgumentError lp_decompose!(c, bp, x, CDF97)
end

# ── pkva_ladder_filter ────────────────────────────────────────────────────────

@testitem "pkva_ladder_filter symmetric 12-tap" tags = [:filters] begin
    f = Contourlets.pkva_ladder_filter()
    @test length(f) == 12
    @test f == reverse(f)   # symmetric
    @test f[6] == 0.63       # pkva12 tap check (f[6] = _PKVA12_HALF[1])
end

# ── conv2d_sep boundary :zero and :periodic via allocating wrapper ────────────

@testitem "conv2d_sep periodic boundary allocating" tags = [:primitives] begin
    using Random
    Random.seed!(95)
    x = randn(8, 8)
    h = [0.25, 0.5, 0.25]
    out = conv2d_sep(x, h, h; boundary = :periodic)
    @test size(out) == size(x)
    @test all(isfinite, out)
end

@testitem "conv2d_sep zero boundary allocating" tags = [:primitives] begin
    using Random
    Random.seed!(96)
    x = randn(8, 8)
    h = [0.25, 0.5, 0.25]
    out = conv2d_sep(x, h, h; boundary = :zero)
    @test size(out) == size(x)
    @test all(isfinite, out)
end

# ── nsdfb_decompose threading kwarg (unused but passed for API symmetry) ──────

@testitem "nsdfb_decompose threading kwarg is accepted" tags = [:directional] begin
    x = randn(16, 16)
    sbs_e = nsdfb_decompose(x, 2, Q2345, 1; threading = Enabled())
    sbs_d = nsdfb_decompose(x, 2, Q2345, 1; threading = Disabled())
    # Results must be identical (threading doesn't affect NSDFB result)
    @test maximum(abs, sbs_e[1] .- sbs_d[1]) < 1.0e-14
end

# ── _extend2 unsupported extmod error ────────────────────────────────────────

@testitem "_extend2 unsupported extmod throws" tags = [:directional] begin
    x = randn(8, 8)
    @test_throws ArgumentError Contourlets._extend2(x, 1, 1, 1, 1, :bad)
end

# ── dfb_reconstruct n=1 (single-subband copy path) ───────────────────────────

@testitem "dfb_reconstruct single subband returns copy" tags = [:directional] begin
    x = randn(16, 16)
    sbs = [x]
    rec = dfb_reconstruct(sbs, Q2345)
    @test rec == x
    @test rec !== x   # must be a copy, not the same object
end

# ── _sefilter2_kernel! sym=true AND threaded=true ─────────────────────────────
# The threaded sym branch is only reached with a symmetric modulated ladder
# filter AND threading=Enabled().  Use the same 3-tap symmetric ladder filter
# from the sym=false test but force threading on.

@testitem "_sefilter2 symmetric filter path threaded" tags = [:directional] begin
    using Random
    Random.seed!(100)
    f_sym = [1.0, 2.0, 1.0]
    qfp_sym = QuincunxFilterPair{Float64}(
        reshape([1.0], 1, 1), reshape([1.0], 1, 1),
        (1, 1), (1, 1), f_sym
    )
    x = randn(32, 32)
    sbs = dfb_decompose(x, 2, qfp_sym; threading = Enabled())
    @test length(sbs) == 4
    rec = dfb_reconstruct(sbs, qfp_sym; threading = Enabled())
    @test size(rec) == size(x)
    @test maximum(abs, rec .- x) < 1.0e-10
end

# ── conv2d! large kernel with threading=Enabled() (FFTW threaded plan) ────────

@testitem "conv2d! large kernel Enabled() threading" tags = [:primitives] begin
    using Random
    Random.seed!(102)
    x = randn(32, 32)
    k = randn(6, 6)   # 36 taps > 25 ⇒ FFTW path
    out_enabled = Contourlets.conv2d(x, k; threading = Enabled())
    out_auto = Contourlets.conv2d(x, k; threading = Auto())
    @test size(out_enabled) == size(x)
    @test maximum(abs, out_enabled .- out_auto) < 1.0e-10
end
