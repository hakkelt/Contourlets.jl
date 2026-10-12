@testitem "QFB odd-width throws ArgumentError" tags = [:directional] begin
    # dir=:col checks n2 of image (columns)
    @test_throws ArgumentError qfb_decompose(randn(8, 7), Q2345; dir = :col)
    # dir=:row transposes, so checks n1 of image (rows)
    @test_throws ArgumentError qfb_decompose(randn(7, 8), Q2345; dir = :row)
    # reconstruct: out has odd n2
    out_odd = zeros(8, 7)
    sb0 = zeros(8, 3); sb1 = zeros(8, 3)
    @test_throws ArgumentError qfb_reconstruct!(out_odd, sb0, sb1, Q2345; dir = :col)
end

@testitem "QFB PR col-direction" tags = [:directional] begin
    using Random
    Random.seed!(20)
    x = randn(32, 32)
    sb0, sb1 = qfb_decompose(x, Q2345)
    @test size(sb0) == (32, 16)
    @test size(sb1) == (32, 16)
    rec = qfb_reconstruct(sb0, sb1, Q2345)
    @test maximum(abs, rec .- x) < 1.0e-12
end

@testitem "QFB PR row-direction" tags = [:directional] begin
    using Random
    Random.seed!(21)
    x = randn(32, 32)
    sb0, sb1 = qfb_decompose(x, Q2345; dir = :row)
    @test size(sb0) == (16, 32)
    rec = qfb_reconstruct(sb0, sb1, Q2345; dir = :row)
    @test maximum(abs, rec .- x) < 1.0e-12
end

@testitem "DFB PR levels 1–4" tags = [:directional] begin
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

@testitem "DFB subband sizes (canonical directional mosaic)" tags = [:directional] begin
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

@testitem "DFB is directionally selective" tags = [:directional] begin
    # An oriented sinusoid concentrates its energy in one of the 8 wedges.
    n = 128
    wave(θ, f) = [cos(2π * f * (cos(θ) * j + sin(θ) * i)) for i in 1:n, j in 1:n]
    for deg in 0:15:179
        e = [sum(abs2, s) for s in dfb_decompose(wave(deg * π / 180, 0.3), 3, Q2345)]
        @test maximum(e) > 0.4 * sum(e)
    end
end

@testitem "DFB level 0 returns single subband" tags = [:directional] begin
    x = randn(8, 8)
    sbs = dfb_decompose(x, 0, Q2345)
    @test length(sbs) == 1
    @test sbs[1] == x
end

@testitem "NSDFB PR levels 1–3, tree_level 1–3" tags = [:directional] begin
    using Random
    Random.seed!(23)
    x = randn(32, 32)
    for l in 1:3, tl in 1:5
        sbs = nsdfb_decompose(x, l, Q2345, tl)
        @test length(sbs) == 2^l
        @test all(s -> size(s) == (32, 32), sbs)
        rec = nsdfb_reconstruct(sbs, Q2345, tl)
        @test maximum(abs, rec .- x) < 1.0e-12
    end
end

@testitem "NSDFB upsamples its filters at coarse pyramid levels" tags = [:directional] begin
    # At tree_level j the filters are Uₖ(z^(2^m I)), m = max(0, j - 2): their
    # response at ω is the base filter's at 2^m ω.  The subbands are zero-phase
    # filterings, so a plane wave of frequency k comes out scaled by Uₖ(2^m k),
    # exactly as the plane wave of frequency 2^m k does at tree_level 1.
    n = 64
    wave(k) = [cos(2π * (k[1] * a + k[2] * b) / n) for a in 0:(n - 1), b in 0:(n - 1)]
    gain(x, s) = sum(s .* x) / sum(abs2, x)
    for (j, m) in ((1, 0), (2, 0), (3, 1), (4, 2), (5, 3)), k in ((1, 2), (3, -1), (2, 2))
        x, xm = wave(k), wave((2^m) .* k)
        coarse = nsdfb_decompose(x, 3, Q2345, j)
        fine = nsdfb_decompose(xm, 3, Q2345, 1)
        @test [gain(x, s) for s in coarse] ≈ [gain(xm, s) for s in fine] atol = 1.0e-12
    end

    # The point of the upsampling: at the coarsest of five pyramid levels the
    # directional subbands separate orientations again.  A sweep of plane waves in
    # that band (radius ≈ n/48 cycles) makes all 8 subbands dominant in turn with
    # the upsampled filters; with the unmodified ones a single subband wins at
    # every angle.
    n = 384
    dominant(j) = Set(
        begin
            x = wave((round(Int, 8 * cos(θ)), round(Int, 8 * sin(θ))))
            argmax([sum(abs2, s) for s in nsdfb_decompose(x, 3, Q2345, j)])
        end for θ in range(0, π; length = 73)[1:(end - 1)]
    )
    @test length(dominant(5)) == 8
    @test length(dominant(1)) <= 2
end

@testitem "NSDFB subbands are directionally selective" tags = [:directional] begin
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

@testitem "NSDFB angle sweep resolves all wedges" tags = [:directional] begin
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

@testitem "DFB matches the Contourlet Toolbox at l ≤ 2" tags = [:directional] begin
    # Output of the MATLAB toolbox's dfbdec_l, run as a black box (see
    # test/reference/README.md).  The toolbox differs from l = 3 on, where it is not
    # the multichannel bank of Do's thesis on a periodic image.
    ref = reinterpret(Float64, read(joinpath(@__DIR__, "..", "reference", "toolbox_dfb_l12.bin")))
    x = [mod(73i^2 + 179j^2 + 31i * j, 1031) / 1031 - 0.5 for i in 1:32, j in 1:32]
    ours = reduce(vcat, [vec(s) for l in 1:2 for s in dfb_decompose(x, l, Q2345)])
    @test length(ours) == length(ref)
    @test maximum(abs, ours .- ref) < 1.0e-13
end

@testitem "DFB channels are filter-then-downsample on a diagonal lattice (Do, eq. 3.23)" tags = [:directional] begin
    # Channel k is yₖ[n] = (hₖ ⊛ x)[Sₖn]: shifting the input by a step of Sₖ's
    # lattice shifts the subband by one sample, exactly, at every depth.
    using Random
    Random.seed!(5)
    x = randn(64, 64)
    for l in 2:5
        sbs = dfb_decompose(x, l, Q2345)
        szs = dfb_subband_sizes(64, 64, l)
        for k in eachindex(sbs)
            S = (64 ÷ szs[k][1], 64 ÷ szs[k][2])
            for step in ((1, 0), (0, 1), (1, 1))
                ys = dfb_decompose(circshift(x, S .* step), l, Q2345)[k]
                @test maximum(abs, ys .- circshift(sbs[k], step)) < 1.0e-12
            end
        end
    end
end

@testitem "NSDFB subbands are the DFB's equivalent filters without decimation" tags = [:directional] begin
    # da Cunha, Zhou & Do: the NSDFB is the DFB with the samplers switched off.  So
    # each DFB subband is its NSDFB subband (scaled by 2^(l/2), up to the channel's
    # delay) sampled on the channel's lattice.
    using Random
    Random.seed!(6)
    x = randn(32, 32)
    for l in 2:4
        d = dfb_decompose(x, l, Q2345)
        ns = nsdfb_decompose(x, l, Q2345)
        szs = dfb_subband_sizes(32, 32, l)
        for k in eachindex(d)
            S = (32 ÷ szs[k][1], 32 ÷ szs[k][2])
            hits = [
                maximum(abs, 2.0^(l / 2) .* sgn .* circshift(ns[k], .-t)[1:S[1]:end, 1:S[2]:end] .- d[k])
                    for t in Iterators.product(-16:16, -16:16), sgn in (1, -1)
            ]
            @test minimum(hits) < 1.0e-12
        end
    end
end

@testitem "DFB and NSDFB resolve every wedge at every depth" tags = [:directional] begin
    # A plane wave sweeping 180° must make each of the 2ˡ subbands dominant in
    # turn.  Perfect reconstruction proves nothing about the partition; this does.
    # (The previous NSDFB, like the NSCT Toolbox's nsdfbdec run as a black box on
    # the same sweep, merged sibling wedges from l = 4 on: 8 of 16 and 16 of 32
    # subbands never dominate.)  Integer frequencies keep the waves periodic on
    # the grid, so no spectral leakage blurs the wedge boundaries.
    #
    # At l = 5 the two 3.6°-wide wedges that border the ±45° fan boundary
    # (subbands 1 and 16) are narrower than the 12-tap fan filters' transition, so
    # a neighbour always outweighs them.  The Contourlet Toolbox's dfbdec_l, run as
    # a black box on the same sweep, misses exactly the same two subbands.
    n = 256
    R = 0.3n
    function wave(θ)
        a, b = round(Int, R * cos(θ)), round(Int, R * sin(θ))
        return [cos(2π * (a * j + b * i) / n) for i in 0:(n - 1), j in 0:(n - 1)]
    end
    for l in 2:5
        angles = range(0, π; length = 8 * 2^l + 1)[1:(end - 1)]
        expected = l == 5 ? setdiff(1:32, (1, 16)) : 1:(2^l)
        for dec in (x -> dfb_decompose(x, l, Q2345), x -> nsdfb_decompose(x, l, Q2345))
            doms = [argmax([sum(abs2, s) for s in dec(wave(θ))]) for θ in angles]
            @test setdiff(expected, doms) == Int[]
        end
    end
end

@testitem "half trees keep the full tree's wedges and reconstruct" tags = [:directional] begin
    # Eslami & Radha's modified banks: one level-1 fan is split, the other kept.
    using Random
    Random.seed!(7)
    x = randn(32, 32)
    for l in 1:4
        m = 2^(l - 1)
        full, nsfull = dfb_decompose(x, l, Q2345), nsdfb_decompose(x, l, Q2345)
        for (h, keep, from) in ((:first, 1:m, 1:m), (:second, 2:(m + 1), (m + 1):(2m)))
            sbs, ns = dfb_decompose(x, l, Q2345; half = h), nsdfb_decompose(x, l, Q2345; half = h)
            @test length(sbs) == length(ns) == m + 1
            @test size.(sbs) == dfb_subband_sizes(32, 32, l; half = h)
            if l > 1
                @test all(sbs[i] ≈ full[j] for (i, j) in zip(keep, from))
                @test all(ns[i] ≈ nsfull[j] for (i, j) in zip(keep, from))
            end
            @test maximum(abs, dfb_reconstruct(sbs, Q2345; half = h) .- x) < 1.0e-12
            @test maximum(abs, nsdfb_reconstruct(ns, Q2345; half = h) .- x) < 1.0e-12
        end
    end
end

@testitem "NSCT is directionally selective for oriented inputs" tags = [:nsct, :directional] begin
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

@testitem "NSCT is shift-invariant" tags = [:nsct, :directional] begin
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

@testitem "QFB in-place decompose col-direction" tags = [:directional] begin
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

@testitem "QFB in-place decompose row-direction" tags = [:directional] begin
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

@testitem "QFB in-place reconstruct col-direction" tags = [:directional] begin
    using Random
    Random.seed!(26)
    x = randn(32, 32)
    sb0, sb1 = qfb_decompose(x, Q2345)
    out = zeros(32, 32)
    qfb_reconstruct!(out, sb0, sb1, Q2345)
    @test maximum(abs, out .- x) < 1.0e-12
end

@testitem "QFB in-place reconstruct row-direction" tags = [:directional] begin
    using Random
    Random.seed!(27)
    x = randn(32, 32)
    sb0, sb1 = qfb_decompose(x, Q2345; dir = :row)
    out = zeros(32, 32)
    qfb_reconstruct!(out, sb0, sb1, Q2345; dir = :row)
    @test maximum(abs, out .- x) < 1.0e-12
end

@testitem "QFB in-place row-direction no extra allocation vs col-direction" tags = [:directional] begin
    using Random
    Random.seed!(28)
    x = randn(32, 32)
    sb0_row = zeros(16, 32); sb1_row = zeros(16, 32); out_row = zeros(32, 32)
    sb0_col = zeros(32, 16); sb1_col = zeros(32, 16); out_col = zeros(32, 32)
    # Warmup both paths
    qfb_decompose!(sb0_row, sb1_row, x, Q2345; dir = :row)
    qfb_reconstruct!(out_row, sb0_row, sb1_row, Q2345; dir = :row)
    qfb_decompose!(sb0_col, sb1_col, x, Q2345; dir = :col)
    qfb_reconstruct!(out_col, sb0_col, sb1_col, Q2345; dir = :col)
    # After warmup: row path must not allocate O(n²) matrix copies.
    # Tolerance of 1024 bytes comfortably absorbs the ~400–500 B of PermutedDimsArray /
    # Adjoint wrapper heap-allocation overhead present in Julia 1.10 (eliminated in 1.11+),
    # while remaining well below one 32×32 Float64 matrix copy (8192 bytes).
    col_dec_alloc = @allocated qfb_decompose!(sb0_col, sb1_col, x, Q2345; dir = :col)
    row_dec_alloc = @allocated qfb_decompose!(sb0_row, sb1_row, x, Q2345; dir = :row)
    @test row_dec_alloc <= col_dec_alloc + 1024
    col_rec_alloc = @allocated qfb_reconstruct!(out_col, sb0_col, sb1_col, Q2345; dir = :col)
    row_rec_alloc = @allocated qfb_reconstruct!(out_row, sb0_row, sb1_row, Q2345; dir = :row)
    @test row_rec_alloc <= col_rec_alloc + 1024
end

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

@testitem "dfb_subband_sizes l=0 is the input size" tags = [:directional] begin
    @test dfb_subband_sizes(64, 64, 0) == [(64, 64)]
end

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

@testitem "the ladder filter must be Phoong's symmetric even-length β" tags = [:directional] begin
    x = randn(32, 32)
    odd = QuincunxFilterPair([0.25, 0.5, 0.25])
    asym = QuincunxFilterPair([0.1, 0.4, 0.5, 0.3])
    for qfp in (odd, asym)
        @test_throws ArgumentError dfb_decompose(x, 2, qfp)
        @test_throws ArgumentError nsdfb_decompose(x, 2, qfp)
    end
end

@testitem "DFB rejects image sizes its sampling lattices do not divide" tags = [:directional] begin
    @test_throws ArgumentError dfb_decompose(randn(30, 32), 3, Q2345)   # 30 not divisible by 4
    @test_throws ArgumentError dfb_decompose(randn(31, 32), 1, Q2345)
    @test_throws ArgumentError dfb_decompose(randn(32, 32), 2, Q2345; half = :neither)
end

@testitem "dfb_reconstruct single subband returns copy" tags = [:directional] begin
    x = randn(16, 16)
    sbs = [x]
    rec = dfb_reconstruct(sbs, Q2345)
    @test rec == x
    @test rec !== x   # must be a copy, not the same object
end

@testitem "DFB and NSDFB on complex data are real-linear" tags = [:directional] begin
    using Random
    Random.seed!(42)
    a, b = randn(32, 32), randn(32, 32)
    z = complex.(a, b)
    for l in 1:3
        dz, da, db = dfb_decompose(z, l, Q2345), dfb_decompose(a, l, Q2345), dfb_decompose(b, l, Q2345)
        @test all(maximum(abs, dz[k] .- complex.(da[k], db[k])) < 1.0e-12 for k in eachindex(dz))
        @test maximum(abs, dfb_reconstruct(dz, Q2345) .- z) < 1.0e-12
        nz, na, nb = nsdfb_decompose(z, l, Q2345), nsdfb_decompose(a, l, Q2345), nsdfb_decompose(b, l, Q2345)
        @test all(maximum(abs, nz[k] .- complex.(na[k], nb[k])) < 1.0e-12 for k in eachindex(nz))
        @test maximum(abs, nsdfb_reconstruct(nz, Q2345) .- z) < 1.0e-12
    end
end
