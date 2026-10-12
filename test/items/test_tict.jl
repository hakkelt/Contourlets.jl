@testitem "TICT and STICT reconstruct exactly" tags = [:tict] begin
    using Random
    Random.seed!(80)
    for (dims, L) in (((32, 32), [2, 1]), ((64, 48), [3, 2, 0]), ((32, 64), [0]))
        p = ContourletParams(J = length(L), L_array = L)
        for x in (randn(dims...), complex.(randn(dims...), randn(dims...)))
            c = tict_forward(x, p)
            @test c isa TICTCoefficients
            @test eltype(c.coarse) == eltype(x)
            @test maximum(abs, tict_inverse(c, p) .- x) < 1.0e-12
            s = stict_forward(x, p)
            @test maximum(abs, stict_inverse(s, p) .- x) < 1.0e-12
        end
    end
    x = randn(Float32, 32, 32)
    p = ContourletParams(J = 2, L_array = [2, 2])
    rec = tict_inverse(tict_forward(x, p), p)
    @test eltype(rec) == Float32
    @test maximum(abs, rec .- x) < 1.0f-4
end

@testitem "TICT and STICT redundancy and subband layout" tags = [:tict] begin
    x = zeros(64, 48)
    p = ContourletParams(J = 3, L_array = [3, 2, 0])
    c = tict_forward(x, p)
    s = stict_forward(x, p)
    # Channels 1 and 4: full tree; 2 and 3: half trees; L = 0: the bare channel.
    @test [length.(lvl) for lvl in c.subbands] == [[8, 5, 5, 8], [4, 3, 3, 4], [1, 1, 1, 1]]
    @test [length.(lvl) for lvl in s.subbands] == [length.(lvl) for lvl in c.subbands]
    total(k) = sum(length, Iterators.flatten(Iterators.flatten(k.subbands))) + length(k.coarse)
    @test all(sb -> size(sb) == size(x), Iterators.flatten(Iterators.flatten(c.subbands)))
    @test total(c) == 45 * length(x)            # 3·(8 + 4) + 2·2 + 4 + 1
    @test total(s) == (4 * 3 + 1) * length(x)   # 4J + 1 (Eslami & Radha §III-C)
end

@testitem "TICT is translation invariant; STICT's pyramid is" tags = [:tict] begin
    using Random
    Random.seed!(81)
    x = randn(64, 64)
    sh = (5, -3)
    p = ContourletParams(J = 3, L_array = [3, 2, 2])
    c, cs = tict_forward(x, p), tict_forward(circshift(x, sh), p)
    @test maximum(abs, circshift(c.coarse, sh) .- cs.coarse) < 1.0e-10
    for (la, lb) in zip(c.subbands, cs.subbands), (ca, cb) in zip(la, lb), (a, b) in zip(ca, cb)
        @test maximum(abs, circshift(a, sh) .- b) < 1.0e-10
    end
    # With no directional splitting the STICT is the TILP alone: translation invariant.
    p0 = ContourletParams(J = 2, L_array = [0, 0])
    s, ss = stict_forward(x, p0), stict_forward(circshift(x, sh), p0)
    for (la, lb) in zip(s.subbands, ss.subbands), (ca, cb) in zip(la, lb)
        @test maximum(abs, circshift(ca[1], sh) .- cb[1]) < 1.0e-10
    end
end

@testitem "TILP is the decimated Laplacian pyramid made translation invariant" tags = [:tict] begin
    # Eslami & Radha: downsampling the level-i channels of the TILP by 2^i gives the
    # polyphase components of the Laplacian pyramid's level-i detail image.  The
    # reference pyramid here is an independent periodic one: filter, keep every
    # other row and column, zero-insert, filter, subtract.
    using FFTW, Random
    Random.seed!(82)
    n = 64
    x = randn(n, n)
    dtft(f, θ) = (c = (length(f) + 1) ÷ 2; sum(f[l] * cis(-θ * (l - c)) for l in eachindex(f)))
    function lp(v)
        m = size(v, 1)
        H = [dtft(CDF97.h, 2π * a / m) * dtft(CDF97.h, 2π * b / m) for a in 0:(m - 1), b in 0:(m - 1)]
        G = [dtft(CDF97.g, 2π * a / m) * dtft(CDF97.g, 2π * b / m) for a in 0:(m - 1), b in 0:(m - 1)]
        coarse = real(ifft(H .* fft(v)))[1:2:end, 1:2:end]
        u = zeros(m, m)
        u[1:2:end, 1:2:end] .= coarse
        return coarse, v .- real(ifft(G .* fft(u)))
    end
    c1, d1 = lp(x)
    c2, d2 = lp(c1)
    tilp = stict_forward(x, ContourletParams(J = 2, L_array = [0, 0]))
    for (c, (kr, kc)) in enumerate(((0, 0), (1, 0), (0, 1), (1, 1)))
        @test tilp.subbands[1][c][1][1:2:end, 1:2:end] ≈ d1[(1 + kr):2:end, (1 + kc):2:end] atol = 1.0e-12
        @test tilp.subbands[2][c][1][1:4:end, 1:4:end] ≈ d2[(1 + kr):2:end, (1 + kc):2:end] atol = 1.0e-12
    end
    @test tilp.coarse[1:4:end, 1:4:end] ≈ lp(c1)[1] atol = 1.0e-12
end

@testitem "TICT/STICT directional stages are the NSDFB/DFB of each channel" tags = [:tict] begin
    using Random
    Random.seed!(83)
    x = randn(64, 64)
    L = [3, 2]
    channels = stict_forward(x, ContourletParams(J = 2, L_array = [0, 0]))
    c = tict_forward(x, ContourletParams(J = 2, L_array = L))
    s = stict_forward(x, ContourletParams(J = 2, L_array = L))
    for i in 1:2, (ch, half) in enumerate((:both, :second, :first, :both))
        w = channels.subbands[i][ch][1]
        @test all(zip(c.subbands[i][ch], nsdfb_decompose(w, L[i], Q2345; half = half))) do (a, b)
            maximum(abs, a .- b) < 1.0e-12
        end
        @test all(zip(s.subbands[i][ch], dfb_decompose(w, L[i], Q2345; half = half))) do (a, b)
            maximum(abs, a .- b) < 1.0e-12
        end
    end
end

@testitem "TICT half trees split the fan their channel's energy is in" tags = [:tict] begin
    # Channel 2 (coset shift along the rows) passes mostly row frequencies and
    # splits that fan, so a row-frequency wave leaves its unsplit fan (subband 1)
    # almost empty; channel 3 is the transpose, with its unsplit fan last.
    n = 64
    wave(k) = [cos(2π * (k[1] * a + k[2] * b) / n) for a in 0:(n - 1), b in 0:(n - 1)]
    p = ContourletParams(J = 1, L_array = [3])
    e(c, ch) = [sum(abs2, s) for s in c.subbands[1][ch]]
    rows = e(tict_forward(wave((24, 3)), p), 2)
    cols = e(tict_forward(wave((3, 24)), p), 3)
    @test rows[1] < 0.01 * sum(rows)
    @test cols[end] < 0.01 * sum(cols)
end

@testitem "TICT/STICT workspaces: allocation-free and equal to the allocating path" tags = [:tict] begin
    using Random
    Random.seed!(84)
    p = ContourletParams(J = 2, L_array = [3, 2])
    for x in (randn(64, 64), complex.(randn(64, 64), randn(64, 64)))
        wt, ws = make_tict_workspace(x, p), make_stict_workspace(x, p)
        @test wt isa TICTWorkspace && ws isa STICTWorkspace
        ct = similar_tict_coefficients(p, size(x); Td = eltype(x))
        cs = similar_stict_coefficients(p, size(x); Td = eltype(x))
        out = similar(x)
        run(th) = (
            tict_forward!(ct, x, p; workspace = wt, threading = th); tict_inverse!(out, ct, p; workspace = wt, threading = th);
            stict_forward!(cs, x, p; workspace = ws, threading = th); stict_inverse!(out, cs, p; workspace = ws, threading = th)
        )
        run(Disabled())
        @test (@allocated run(Disabled())) == 0
        @test maximum(abs, out .- x) < 1.0e-12
        # Threaded, `@batch` boxes the captured workspace: a few hundred bytes per call.
        run(Enabled())
        @test (@allocated run(Enabled())) < 1000
        @test maximum(abs, out .- x) < 1.0e-12
        ref = tict_forward(x, p)
        tict_forward!(ct, x, p; workspace = wt)
        @test all(a == b for (la, lb) in zip(ct.subbands, ref.subbands) for (ca, cb) in zip(la, lb) for (a, b) in zip(ca, cb))
        refs = stict_forward(x, p)
        stict_forward!(cs, x, p; workspace = ws)
        @test all(a == b for (la, lb) in zip(cs.subbands, refs.subbands) for (ca, cb) in zip(la, lb) for (a, b) in zip(ca, cb))
    end
end

@testitem "TICT/STICT argument checks" tags = [:tict] begin
    p = ContourletParams(J = 1, L_array = [2])
    ws = make_tict_workspace(p, (32, 32))
    @test_throws DimensionMismatch tict_forward(zeros(16, 16), p; workspace = ws)
    # The `!` forms need the workspace's element type.
    c32 = similar_tict_coefficients(p, (32, 32); Td = Float32)
    @test_throws ArgumentError tict_forward!(c32, zeros(32, 32), p; workspace = ws)
    @test_throws ArgumentError tict_inverse!(zeros(Float32, 32, 32), tict_forward(zeros(32, 32), p), p; workspace = ws)
    sws = make_stict_workspace(p, (32, 32))
    @test_throws ArgumentError stict_inverse(similar_stict_coefficients(p, (32, 32); Td = Float32), p; workspace = sws)
    haar = QuincunxFilterPair([0.5 0.5], [1.0 1.0], (1, 1), (1, 2))
    @test_throws ArgumentError tict_forward(zeros(32, 32), ContourletParams(J = 1, L_array = [2], dfb_filters = haar))
    @test_throws ArgumentError stict_forward(zeros(20, 20), ContourletParams(J = 1, L_array = [4]))
end
