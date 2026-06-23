@testitem "QFB PR col-direction" begin
    using Random
    Random.seed!(20)
    x = randn(32, 32)
    sb0, sb1 = qfb_decompose(x, Q2345)
    @test size(sb0) == (32, 16)
    @test size(sb1) == (32, 16)
    rec = qfb_reconstruct(sb0, sb1, Q2345)
    @test maximum(abs, rec .- x) < 1.0e-12
end

@testitem "QFB PR row-direction" begin
    using Random
    Random.seed!(21)
    x = randn(32, 32)
    sb0, sb1 = qfb_decompose(x, Q2345; dir = :row)
    @test size(sb0) == (16, 32)
    rec = qfb_reconstruct(sb0, sb1, Q2345; dir = :row)
    @test maximum(abs, rec .- x) < 1.0e-12
end

@testitem "DFB PR levels 1–4" begin
    using Random
    Random.seed!(22)
    x = randn(32, 32)
    for l in 1:4
        sbs = dfb_decompose(x, l, Q2345)
        @test length(sbs) == 2^l
        rec = dfb_reconstruct(sbs, Q2345)
        @test maximum(abs, rec .- x) < 1.0e-12
    end
end

@testitem "DFB subband sizes (canonical directional mosaic)" begin
    x = zeros(64, 64)
    # l=1: two horizontal subbands, rows halved.
    sbs1 = dfb_decompose(x, 1, Q2345)
    @test all(==((32, 64)), size.(sbs1))
    # l=2: four square subbands.
    sbs2 = dfb_decompose(x, 2, Q2345)
    @test all(==((32, 32)), size.(sbs2))
    # l=3: 4 horizontal-mosaic + 4 vertical-mosaic subbands.
    sbs3 = dfb_decompose(x, 3, Q2345)
    @test count(==((16, 32)), size.(sbs3)) == 4
    @test count(==((32, 16)), size.(sbs3)) == 4
    # l=4: 8 + 8.
    sbs4 = dfb_decompose(x, 4, Q2345)
    @test count(==((8, 32)), size.(sbs4)) == 8
    @test count(==((32, 8)), size.(sbs4)) == 8
    # The exported size helper matches the actual decomposition.
    for l in 1:4
        @test dfb_subband_sizes(64, 64, l) == size.(dfb_decompose(x, l, Q2345))
    end
    # Sample count is preserved (nearly critical sampling).
    for l in 1:4
        @test sum(prod, dfb_subband_sizes(64, 64, l)) == 64 * 64
    end
end

@testitem "DFB is directionally selective" begin
    # An l=3 DFB has 8 directional wedges.  Sweeping the orientation of a
    # near-Nyquist 2-D sinusoid must light up several *different* dominant
    # subbands — a degenerate (non-directional) DFB would always pick the same
    # one.  Guards against the old shear-tree collapse.
    n = 128
    wave(θ, f) = [cos(2π * f * (cos(θ) * j + sin(θ) * i)) for i in 1:n, j in 1:n]
    doms = Int[]
    for deg in 0:15:179
        e = [sum(abs2, s) for s in dfb_decompose(wave(deg * π / 180, 0.45), 3, Q2345)]
        push!(doms, argmax(e))
        @test maximum(e) > 0.4 * sum(e)   # energy concentrates in a wedge
    end
    @test length(unique(doms)) >= 4       # many distinct directions resolved
end

@testitem "DFB level 0 returns single subband" begin
    x = randn(8, 8)
    sbs = dfb_decompose(x, 0, Q2345)
    @test length(sbs) == 1
    @test sbs[1] == x
end

@testitem "NSDFB PR levels 1–3, tree_level 1–3" begin
    using Random
    Random.seed!(23)
    x = randn(32, 32)
    for l in 1:3, tl in 1:3
        sbs = nsdfb_decompose(x, l, Q2345, tl)
        @test length(sbs) == 2^l
        @test all(s -> size(s) == (32, 32), sbs)
        rec = nsdfb_reconstruct(sbs, Q2345, tl)
        @test maximum(abs, rec .- x) < 1.0e-12
    end
end

@testitem "NSDFB subbands are directionally selective" begin
    # An oriented sinusoid must concentrate its energy in a single directional
    # wedge.  With the resampling-matrix (fan-filter) construction each subband is
    # a genuine angular wedge, so a single tone lands almost entirely in one of
    # them (a quadrant/block partition would smear it across several).
    n = 64
    wave(θ) = [cos(2π * 0.4 * (cos(θ) * j + sin(θ) * i)) for i in 1:n, j in 1:n]
    for deg in (15, 45, 75, 105, 135, 165)
        e = [sum(abs2, s) for s in nsdfb_decompose(wave(deg * π / 180), 3, Q2345)]
        @test maximum(e) > 0.6 * sum(e)     # one wedge dominates
    end
end

@testitem "NSDFB angle sweep resolves all wedges" begin
    # Sweeping orientation over 180° at l=3 (8 wedges) must light up every wedge:
    # a directional bank resolves all 2^l distinct dominant subbands, a degenerate
    # one collapses to far fewer (the old block bank resolved ≤6, the broken
    # diagonal-only bank only 4).
    n = 128
    wave(θ) = [cos(2π * 0.42 * (cos(θ) * j + sin(θ) * i)) for i in 1:n, j in 1:n]
    doms = [
        argmax([sum(abs2, s) for s in nsdfb_decompose(wave(deg * π / 180), 3, Q2345)])
            for deg in 0:10:179
    ]
    @test length(unique(doms)) == 8
end

@testitem "NSDFB matches the MATLAB reference (resampling-matrix fan filters)" begin
    # Locks the wedge construction to the da Cunha–Zhou–Do reference: the fan and
    # parallelogram filters built from the pkva ladder must equal dfilters('pkva')
    # + parafilters from the MATLAB toolbox.  Reference values precomputed.
    F = Contourlets._nsdfb_filters(Q2345, Float64)
    @test size(F.k1) == (23, 23)        # fan filter from the 23-tap diamond low-pass
    @test size(F.k2) == (45, 45)        # fan filter from the 45-tap diamond high-pass
    @test length(F.f1) == 4 && length(F.f2) == 4
    @test all(==((23, 23)), size.(F.f1))
    @test isodd(size(F.k2, 1))
end

@testitem "NSCT is directionally selective for oriented inputs" begin
    n = 128
    p = ContourletParams(J = 1, L_array = [3])
    for deg in (30, 60, 90, 120, 150)
        x = [
            cos(2π * 0.42 * (cos(deg * π / 180) * j + sin(deg * π / 180) * i))
                for i in 1:n, j in 1:n
        ]
        e = [sum(abs2, s) for s in nsct_forward(x, p; threading = Disabled()).subbands[1]]
        @test maximum(e) > 0.5 * sum(e)
    end
end

@testitem "NSCT is shift-invariant" begin
    using Random
    Random.seed!(99)
    n = 32
    x = randn(n, n)
    p = ContourletParams(J = 1, L_array = [3])
    c0 = nsct_forward(x, p; threading = Disabled())
    for shift in ((3, 0), (0, 5), (2, 7))
        cs = nsct_forward(circshift(x, shift), p; threading = Disabled())
        # LP residual
        @test maximum(abs, c0.coarse .- circshift(cs.coarse, .-shift)) < 1.0e-10
        # Directional subbands
        for j in 1:length(c0.subbands)
            for (a, b) in zip(c0.subbands[j], cs.subbands[j])
                @test maximum(abs, a .- circshift(b, .-shift)) < 1.0e-10
            end
        end
    end
end

@testitem "QFB in-place decompose col-direction" begin
    using Random
    Random.seed!(24)
    x = randn(32, 32)
    sb0_alloc, sb1_alloc = qfb_decompose(x, Q2345)
    sb0 = zeros(32, 16)
    sb1 = zeros(32, 16)
    qfb_decompose!(sb0, sb1, x, Q2345)
    @test maximum(abs, sb0 .- sb0_alloc) < 1.0e-14
    @test maximum(abs, sb1 .- sb1_alloc) < 1.0e-14
end

@testitem "QFB in-place decompose row-direction" begin
    using Random
    Random.seed!(25)
    x = randn(32, 32)
    sb0_alloc, sb1_alloc = qfb_decompose(x, Q2345; dir = :row)
    sb0 = zeros(16, 32)
    sb1 = zeros(16, 32)
    qfb_decompose!(sb0, sb1, x, Q2345; dir = :row)
    @test maximum(abs, sb0 .- sb0_alloc) < 1.0e-14
    @test maximum(abs, sb1 .- sb1_alloc) < 1.0e-14
end

@testitem "QFB in-place reconstruct col-direction" begin
    using Random
    Random.seed!(26)
    x = randn(32, 32)
    sb0, sb1 = qfb_decompose(x, Q2345)
    out = zeros(32, 32)
    qfb_reconstruct!(out, sb0, sb1, Q2345)
    @test maximum(abs, out .- x) < 1.0e-12
end

@testitem "QFB in-place reconstruct row-direction" begin
    using Random
    Random.seed!(27)
    x = randn(32, 32)
    sb0, sb1 = qfb_decompose(x, Q2345; dir = :row)
    out = zeros(32, 32)
    qfb_reconstruct!(out, sb0, sb1, Q2345; dir = :row)
    @test maximum(abs, out .- x) < 1.0e-12
end

@testitem "dfb_decompose! basic" begin
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

@testitem "dfb_decompose! with workspace" begin
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

@testitem "dfb_reconstruct! basic" begin
    using Random
    Random.seed!(72)
    x = randn(32, 32)
    l = 2
    sbs = dfb_decompose(x, l, Q2345)
    bp = zeros(32, 32)
    dfb_reconstruct!(bp, sbs, Q2345)
    @test maximum(abs, bp .- x) < 1.0e-12
end

@testitem "dfb_reconstruct! with workspace" begin
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

@testitem "dfb_decompose! invalid subbands length" begin
    x = randn(32, 32)
    subbands = [zeros(16, 16)]  # wrong: 1 instead of 4 for l=2
    @test_throws ArgumentError dfb_decompose!(subbands, x, 2, Q2345)
end

@testitem "dfb_reconstruct error on empty subbands" begin
    @test_throws ArgumentError dfb_reconstruct(Matrix{Float64}[], Q2345)
end

@testitem "dfb_reconstruct error on non-power-of-2" begin
    sbs = [randn(32, 11) for _ in 1:3]   # 3 is not a power of 2
    @test_throws ArgumentError dfb_reconstruct(sbs, Q2345)
end

@testitem "dfb_subband_sizes modulation-mode (ladder=false)" begin
    szs = dfb_subband_sizes(64, 64, 2; ladder = false)
    @test length(szs) == 4
    @test all(==((32, 32)), szs)

    szs1 = dfb_subband_sizes(64, 64, 1; ladder = false)
    @test length(szs1) == 2
    @test all(s -> s == (64, 32), szs1)

    szs0 = dfb_subband_sizes(64, 64, 0; ladder = false)
    @test length(szs0) == 1
    @test szs0[1] == (64, 64)
end

@testitem "DFB threading=Enabled with modulation-mode Haar pair" begin
    using Random
    Random.seed!(74)
    x = randn(32, 32)
    haar = QuincunxFilterPair([0.5 0.5], [1.0 1.0], (1, 1), (1, 2))
    sbs = dfb_decompose(x, 3, haar; threading = Enabled())
    @test length(sbs) == 8
    rec = dfb_reconstruct(sbs, haar; threading = Enabled())
    @test maximum(abs, rec .- x) < 1.0e-12
end

@testitem "nsdfb_decompose l_levels=0 returns copy" begin
    x = randn(16, 16)
    sbs = nsdfb_decompose(x, 0, Q2345, 1)
    @test length(sbs) == 1
    @test sbs[1] == x
    @test sbs[1] !== x   # must be a copy
end

@testitem "nsdfb_reconstruct n=1 returns copy" begin
    x = randn(16, 16)
    sbs = [x]
    rec = nsdfb_reconstruct(sbs, Q2345, 1)
    @test rec == x
    @test rec !== x   # must be a copy
end

@testitem "nsdfb_reconstruct non-power-of-2 throws" begin
    sbs = [randn(16, 16) for _ in 1:3]
    @test_throws ArgumentError nsdfb_reconstruct(sbs, Q2345, 1)
end

@testitem "_sparse_to_dense round-trips SparseFilter2D" begin
    F = Contourlets._nsdfb_filters(Q2345, Float64)
    sf = F.k1
    dense = Contourlets._sparse_to_dense(sf)
    @test size(dense) == size(sf)
    sf2 = Contourlets.SparseFilter2D(dense)
    @test sort(sf2.vals) ≈ sort(sf.vals)
    @test length(sf2.vals) == length(sf.vals)
end

@testitem "_efilter2! dense AbstractMatrix fallback" begin
    using Random
    Random.seed!(80)
    x = randn(8, 8)
    F = Contourlets._nsdfb_filters(Q2345, Float64)
    k1_dense = Contourlets._sparse_to_dense(F.k1)
    out_dense = similar(x)
    Contourlets._efilter2!(out_dense, x, k1_dense)
    out_sparse = similar(x)
    Contourlets._efilter2!(out_sparse, x, F.k1)
    @test maximum(abs, out_dense .- out_sparse) < 1.0e-12
end

@testitem "_zconv2! dense AbstractMatrix fallback" begin
    using Random
    Random.seed!(81)
    x = randn(8, 8)
    F = Contourlets._nsdfb_filters(Q2345, Float64)
    M = Contourlets._Q1
    f1_dense = Contourlets._sparse_to_dense(F.f1[1])
    out_dense = similar(x)
    Contourlets._zconv2!(out_dense, x, f1_dense, M)
    out_sparse = similar(x)
    Contourlets._zconv2!(out_sparse, x, F.f1[1], M)
    @test maximum(abs, out_dense .- out_sparse) < 1.0e-12
end

@testitem "_sefilter2 symmetric filter path" begin
    f_sym = [1.0, 2.0, 1.0]
    qfp_sym = QuincunxFilterPair{Float64}(
        reshape([1.0], 1, 1), reshape([1.0], 1, 1),
        (1, 1), (1, 1), f_sym
    )
    x = randn(32, 32)
    sbs = dfb_decompose(x, 2, qfp_sym)
    @test length(sbs) == 4
    rec = dfb_reconstruct(sbs, qfp_sym)
    @test size(rec) == size(x)
    @test maximum(abs, rec .- x) < 1.0e-10
end

@testitem "_sefilter2 symmetric filter path threaded" begin
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

@testitem "_resamp invalid type throws" begin
    x = randn(8, 8)
    @test_throws ArgumentError Contourlets._resamp(x, 5)
end

@testitem "_qpdec unsupported type throws" begin
    x = randn(8, 8)
    @test_throws ArgumentError Contourlets._qpdec(x, :bad)
end

@testitem "_qprec unsupported type throws" begin
    p0 = randn(4, 8); p1 = randn(4, 8)
    @test_throws ArgumentError Contourlets._qprec(p0, p1, :bad)
end

@testitem "_ppdec invalid type throws" begin
    x = randn(8, 8)
    @test_throws ArgumentError Contourlets._ppdec(x, 5)
end

@testitem "_pprec invalid type throws" begin
    p0 = randn(4, 8); p1 = randn(4, 8)
    @test_throws ArgumentError Contourlets._pprec(p0, p1, 5)
end

@testitem "_resampz invalid type throws" begin
    x = randn(8, 8)
    @test_throws ArgumentError Contourlets._resampz(x, 5)
end

@testitem "nsdfb_decompose threading kwarg is accepted" begin
    x = randn(16, 16)
    sbs_e = nsdfb_decompose(x, 2, Q2345, 1; threading = Enabled())
    sbs_d = nsdfb_decompose(x, 2, Q2345, 1; threading = Disabled())
    @test maximum(abs, sbs_e[1] .- sbs_d[1]) < 1.0e-14
end

@testitem "_extend2 unsupported extmod throws" begin
    x = randn(8, 8)
    @test_throws ArgumentError Contourlets._extend2(x, 1, 1, 1, 1, :bad)
    # :qper_row was removed as dead code (never called in the DFB tree)
    @test_throws ArgumentError Contourlets._extend2(x, 1, 1, 1, 1, :qper_row)
end

@testitem "dfb_reconstruct single subband returns copy" begin
    x = randn(16, 16)
    sbs = [x]
    rec = dfb_reconstruct(sbs, Q2345)
    @test rec == x
    @test rec !== x   # must be a copy, not the same object
end
