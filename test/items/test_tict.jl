@testitem "TICT and STICT reconstruct exactly" tags = [:tict] begin
    using Random
    Random.seed!(80)
    for (dims, L) in (((32, 32), [2, 1]), ((64, 48), [3, 2, 0]), ((32, 64), [0]))
        for x in (randn(dims...), complex.(randn(dims...), randn(dims...)))
            t = TICT(direction_levels = L)
            c = analyze(t, x)
            @test c isa Array{eltype(x), 3}
            @test maximum(abs, synthesize(t, c) .- x) < 1.0e-12
            st = STICT(direction_levels = L)
            s = analyze(st, x)
            @test s isa SubbandVector{eltype(x)}
            @test maximum(abs, synthesize(st, s) .- x) < 1.0e-12
        end
    end
    x = randn(Float32, 32, 32)
    t = TICT(scales = 2, direction_levels = [2, 2])
    rec = synthesize(t, analyze(t, x))
    @test eltype(rec) == Float32
    @test maximum(abs, rec .- x) < 1.0f-4
end

@testitem "TICT and STICT redundancy and subband layout" tags = [:tict] begin
    x = zeros(64, 48)
    L = [3, 2, 0]
    t, st = TICT(scales = 3, direction_levels = L), STICT(scales = 3, direction_levels = L)
    c = analyze(t, x)
    s = analyze(st, x)
    layout = subband_layout(t, size(x))
    @test layout == subband_layout(st, size(x))
    @test layout[1] == (level = 4, channel = 0, direction = 0)
    # Channels 1 and 4: full tree; 2 and 3: half trees; L = 0: the bare channel.
    @test [count(b -> b.level == i && b.channel == ch, layout) for i in 1:3, ch in 1:4] ==
        [8 5 5 8; 4 3 3 4; 1 1 1 1]
    @test all(i -> size(subband(t, c, i)) == size(x), 1:nsubbands(t, size(x)))
    @test length(c) == 45 * length(x)            # 3·(8 + 4) + 2·2 + 4 + 1
    @test length(s) == (4 * 3 + 1) * length(x)   # 4J + 1 (Eslami & Radha §III-C)
end

@testitem "TICT is translation invariant; STICT's pyramid is" tags = [:tict] begin
    using Random
    Random.seed!(81)
    x = randn(64, 64)
    sh = (5, -3)
    t = TICT(scales = 3, direction_levels = [3, 2, 2])
    c, cs = analyze(t, x), analyze(t, circshift(x, sh))
    for i in axes(c, 3)
        @test maximum(abs, circshift(subband(t, c, i), sh) .- subband(t, cs, i)) < 1.0e-10
    end
    # With no directional splitting the STICT is the TILP alone: translation invariant.
    t0 = STICT(scales = 2, direction_levels = [0, 0])
    s, ss = analyze(t0, x), analyze(t0, circshift(x, sh))
    for i in 2:nsubbands(t0, size(x))
        @test maximum(abs, circshift(subband(t0, s, i), sh) .- subband(t0, ss, i)) < 1.0e-10
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
    t = STICT(scales = 2, direction_levels = [0, 0])
    tilp = analyze(t, x)
    # With L = 0, subband 1 + 4(i - 1) + c is channel c of level i.
    channel(i, c) = subband(t, tilp, 1 + 4 * (i - 1) + c)
    for (c, (kr, kc)) in enumerate(((0, 0), (1, 0), (0, 1), (1, 1)))
        @test channel(1, c)[1:2:end, 1:2:end] ≈ d1[(1 + kr):2:end, (1 + kc):2:end] atol = 1.0e-12
        @test channel(2, c)[1:4:end, 1:4:end] ≈ d2[(1 + kr):2:end, (1 + kc):2:end] atol = 1.0e-12
    end
    @test subband(t, tilp, 1)[1:4:end, 1:4:end] ≈ lp(c1)[1] atol = 1.0e-12
end

@testitem "TICT/STICT directional stages are the NSDFB/DFB of each channel" tags = [:tict] begin
    using Random
    Random.seed!(83)
    x = randn(64, 64)
    L = [3, 2]
    t0 = STICT(scales = 2, direction_levels = [0, 0])
    channels = analyze(t0, x)
    t, st = TICT(scales = 2, direction_levels = L), STICT(scales = 2, direction_levels = L)
    c, s = analyze(t, x), analyze(st, x)
    layout = subband_layout(t, size(x))
    for i in 1:2, (ch, half) in enumerate((:both, :second, :first, :both))
        w = subband(t0, channels, 1 + 4 * (i - 1) + ch)
        idx = findall(b -> b.level == i && b.channel == ch, layout)
        @test all(zip(idx, Contourlets.nsdfb_decompose(w, L[i], Q2345; half = half))) do (k, b)
            maximum(abs, subband(t, c, k) .- b) < 1.0e-12
        end
        @test all(zip(idx, Contourlets.dfb_decompose(w, L[i], Q2345; half = half))) do (k, b)
            maximum(abs, subband(st, s, k) .- b) < 1.0e-12
        end
    end
end

@testitem "TICT half trees split the fan their channel's energy is in" tags = [:tict] begin
    # Channel 2 (coset shift along the rows) passes mostly row frequencies and
    # splits that fan, so a row-frequency wave leaves its unsplit fan (subband 1)
    # almost empty; channel 3 is the transpose, with its unsplit fan last.
    n = 64
    wave(k) = [cos(2π * (k[1] * a + k[2] * b) / n) for a in 0:(n - 1), b in 0:(n - 1)]
    t = TICT(scales = 1, direction_levels = [3])
    layout = subband_layout(t, (n, n))
    function e(c, ch)
        idx = findall(b -> b.channel == ch, layout)
        return [sum(abs2, subband(t, c, k)) for k in idx]
    end
    rows = e(analyze(t, wave((24, 3))), 2)
    cols = e(analyze(t, wave((3, 24))), 3)
    @test rows[1] < 0.01 * sum(rows)
    @test cols[end] < 0.01 * sum(cols)
end

@testitem "TICT/STICT workspaces: allocation-free and equal to the allocating path" tags = [:tict] begin
    using Random
    Random.seed!(84)
    for x in (randn(64, 64), complex.(randn(64, 64), randn(64, 64)))
        t, st = TICT(scales = 2, direction_levels = [3, 2]), STICT(scales = 2, direction_levels = [3, 2])
        wt, ws = make_workspace(t, x), make_workspace(st, x)
        @test wt isa TICTWorkspace && ws isa STICTWorkspace
        ct = similar_coefficients(t, x)
        cs = similar_coefficients(st, x)
        out = similar(x)
        run(th) = (
            analyze!(ct, t, x; workspace = wt, threaded = th); synthesize!(out, t, ct; workspace = wt, threaded = th);
            analyze!(cs, st, x; workspace = ws); synthesize!(out, st, cs; workspace = ws)
        )
        run(false)
        @test (@allocated run(false)) == 0
        @test maximum(abs, out .- x) < 1.0e-12
        # Threaded, `@batch` boxes the captured workspace: a few hundred bytes per
        # call.  Where it may not run (a second interactive thread, as in some test
        # runners) the loops fall back to `Threads.@threads`: a few KB.
        run(true)
        bound = Contourlets.MultiscaleFilterBanks._batch_safe() ? 1000 : 8192
        @test (@allocated run(true)) < bound
        @test maximum(abs, out .- x) < 1.0e-12
        analyze!(ct, t, x; workspace = wt)
        @test ct == analyze(t, x)
        analyze!(cs, st, x; workspace = ws)
        @test cs == analyze(st, x)
    end
end

@testitem "TICT stages views and transposes through the workspace" tags = [:tict] begin
    using Random
    Random.seed!(42)
    t = TICT(scales = 2, direction_levels = [2, 1])
    big = randn(40, 40)
    x = view(big, 3:34, 5:36)
    xd = collect(x)
    ref = analyze(t, xd)
    same(c, r) = maximum(abs, c .- r) < 1.0e-12
    @test same(analyze(t, x), ref)
    @test same(analyze(t, transpose(permutedims(xd))), ref)
    ws = make_workspace(t, Float64, (32, 32))
    c = similar_coefficients(t, xd)
    analyze!(c, t, x; workspace = ws)
    @test same(c, ref)
    out = zeros(40, 40)
    synthesize!(view(out, 3:34, 5:36), t, ref; workspace = ws)
    @test maximum(abs, view(out, 3:34, 5:36) .- xd) < 1.0e-12
    @test all(iszero, view(out, 1:2, :))
    # A dense input still allocates nothing.
    let c = c, xd = xd, t = t, ws = ws
        run() = analyze!(c, t, xd; workspace = ws, threaded = false)
        run()
        @test (@allocated run()) == 0
    end
end

@testitem "TICT/STICT argument checks" tags = [:tict] begin
    t = TICT(scales = 1, direction_levels = [2])
    ws = make_workspace(t, Float64, (32, 32))
    @test_throws DimensionMismatch analyze(t, zeros(16, 16); workspace = ws)
    # The `!` forms need the workspace's element type.
    c32 = similar_coefficients(t, zeros(Float32, 32, 32))
    @test_throws ArgumentError analyze!(c32, t, zeros(32, 32); workspace = ws)
    @test_throws ArgumentError synthesize!(zeros(Float32, 32, 32), t, analyze(t, zeros(32, 32)); workspace = ws)
    st = STICT(scales = 1, direction_levels = [2])
    sws = make_workspace(st, Float64, (32, 32))
    @test_throws ArgumentError synthesize(st, similar_coefficients(st, zeros(Float32, 32, 32)); workspace = sws)
    @test_throws ArgumentError analyze(STICT(scales = 1, direction_levels = [4]), zeros(20, 20))
end
