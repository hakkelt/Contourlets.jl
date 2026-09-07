@testitem "nsdfb3d — subband counting and index round trip" tags = [:directional] begin
    for L in 0:3
        fb = make_nsdfb3d((8, 8, 8), Float64; levels = L, order = 4)
        @test nsdfb3d_nsubbands(fb) == 3 * 2^L
        na, nb = 1 << fb.levels[1], 1 << fb.levels[2]
        @test na * nb == 2^L
        seen = Int[]
        for i in 1:3, p in 1:na, q in 1:nb
            idx = nsdfb3d_subband_index(fb, i, p, q)
            push!(seen, idx)
            @test nsdfb3d_subband_triple(fb, idx) == (i, p, q)
        end
        @test sort(seen) == collect(1:nsdfb3d_nsubbands(fb))
    end
    fb = make_nsdfb3d((8, 8, 8), Float64; levels = 2, order = 4)
    @test_throws ArgumentError nsdfb3d_subband_index(fb, 4, 1, 1)
    @test_throws ArgumentError nsdfb3d_subband_index(fb, 1, 99, 1)
    @test_throws ArgumentError nsdfb3d_subband_triple(fb, 0)
    @test_throws ArgumentError make_nsdfb3d((8, 8, 8), Float64; levels = -1)
end

@testitem "nsdfb3d — weights are a partition of unity" tags = [:directional] begin
    for L in 0:3
        fb = make_nsdfb3d((10, 12, 8), Float64; levels = L, order = 6)
        total = sum(nsdfb3d_weight(fb, i) for i in 1:nsdfb3d_nsubbands(fb))
        @test maximum(abs, total .- 1) < 1.0e-13
        # Tight by default, so it is the squares of the masks that sum to 1.
        squares = sum(nsdfb3d_mask(fb, i) .^ 2 for i in 1:nsdfb3d_nsubbands(fb))
        @test maximum(abs, squares .- 1) < 1.0e-13
    end
end

@testitem "nsdfb3d — tight masks are a Parseval frame" tags = [:directional] begin
    using Random
    x = randn(Xoshiro(81), 16, 16, 16)
    for L in 0:3
        fb = make_nsdfb3d(size(x), Float64; levels = L, order = 6)
        s = nsdfb3d_decompose(x, fb)
        @test abs(sum(sum(abs2, b) for b in s) / sum(abs2, x) - 1) < 1.0e-12
        @test maximum(abs, nsdfb3d_reconstruct(s, fb) .- x) < 1.0e-12
    end
    loose = make_nsdfb3d(size(x), Float64; levels = 2, order = 6, tight = false)
    sl = nsdfb3d_decompose(x, loose)
    @test sum(sum(abs2, b) for b in sl) / sum(abs2, x) < 0.9
    @test maximum(abs, nsdfb3d_reconstruct(sl, loose) .- x) < 1.0e-12
end

@testitem "nsdfb3d — masks are even in frequency" tags = [:directional] begin
    # A wedge must give the same value to ω and -ω, or the masked spectrum is
    # not Hermitian and the real subband loses information. On the DFT grid the
    # trap is that an even-length axis fixes its Nyquist bin under ω -> -ω.
    # 12 has `2π(n/2)/n == π` exactly; 26 rounds a ulp *past* π (so folding sends
    # the Nyquist bin to ≈ -π) and 30 a ulp short of it. Identifying that bin by
    # value rather than by index misses both, and the tight bank then silently
    # loses perfect reconstruction — so all three sizes are pinned here.
    for n in (12, 26, 30)
        fb = make_nsdfb3d((n, n, n), Float64; levels = 2, order = 6)
        neg(i) = mod(1 - i, n) + 1
        for idx in 1:nsdfb3d_nsubbands(fb)
            m = nsdfb3d_mask(fb, idx)
            # On the two self-conjugate planes of the rfft grid the mask has to
            # be even in the remaining two axes on its own.
            for i in (1, n ÷ 2 + 1), j in 1:n, k in 1:n
                @test m[i, j, k] ≈ m[i, neg(j), neg(k)] atol = 1.0e-13
            end
        end
    end
end

@testitem "nsdfb3d — tight PR survives a Nyquist bin that is not exactly π" tags = [:directional] begin
    using Random
    # `dft_frequencies` computes `2π(k-1)/n`, which lands an ulp off `π` at the
    # Nyquist bin for about half of all even `n`. The wedge masks must still be
    # even there, or the `√W` synthesis of `tight = true` is lossy.
    for dims in ((30, 16, 16), (16, 30, 16), (16, 16, 30), (26, 26, 26), (60, 16, 16))
        x = randn(Xoshiro(83), dims...)
        for L in 1:3
            fb = make_nsdfb3d(dims, Float64; levels = L, order = 6)
            s = nsdfb3d_decompose(x, fb)
            @test maximum(abs, nsdfb3d_reconstruct(s, fb) .- x) < 1.0e-11
            @test abs(sum(sum(abs2, b) for b in s) / sum(abs2, x) - 1) < 1.0e-11
        end
    end
end

@testitem "nsdfb3d — wedges of one hourglass sum to that hourglass weight" tags = [:directional] begin
    fb = make_nsdfb3d((12, 12, 12), Float64; levels = 3, order = 6)
    na, nb = 1 << fb.levels[1], 1 << fb.levels[2]
    for i in 1:3
        acc = sum(
            nsdfb3d_weight(fb, nsdfb3d_subband_index(fb, i, p, q))
                for p in 1:na, q in 1:nb
        )
        @test maximum(abs, acc .- hourglass3d_weight(fb.fb, i)) < 1.0e-13
    end
end

@testitem "nsdfb3d — perfect reconstruction" tags = [:directional] begin
    using Random
    Random.seed!(80)
    for dims in ((16, 16, 16), (12, 20, 8))
        x = randn(dims...)
        for L in 0:3, tr in (0.0, 0.5, 1.0)
            fb = make_nsdfb3d(dims, Float64; levels = L, order = 6, transition = tr)
            s = nsdfb3d_decompose(x, fb)
            @test length(s) == nsdfb3d_nsubbands(fb)
            @test maximum(abs, nsdfb3d_reconstruct(s, fb) .- x) < 1.0e-12
        end
    end
end

@testitem "nsdfb3d — plane waves land in the predicted wedge" tags = [:directional] begin
    n = 32
    fb = make_nsdfb3d((n, n, n), Float64; levels = 4, order = 12, transition = 0.5)
    la, lb = fb.levels
    transverse(i) = i == 1 ? (2, 3) : (i == 2 ? (1, 3) : (1, 2))
    function predict(k)
        i = argmax(abs.(k))
        a, b = transverse(i)
        p = clamp(floor(Int, (k[a] / k[i] + 1) / (2 / (1 << la))) + 1, 1, 1 << la)
        q = clamp(floor(Int, (k[b] / k[i] + 1) / (2 / (1 << lb))) + 1, 1, 1 << lb)
        return nsdfb3d_subband_index(fb, i, p, q)
    end
    # Frequencies chosen away from wedge and hourglass boundaries: no two
    # magnitudes equal, and neither ratio lands on a wedge edge.
    for k in ([2, 5, 9], [5, 9, -12], [-2, 3, -13], [-13, -9, 7], [9, 7, -12], [3, 11, -7])
        v = [
            cospi(2 * (k[1] * (I[1] - 1) + k[2] * (I[2] - 1) + k[3] * (I[3] - 1)) / n)
                for I in CartesianIndices((n, n, n))
        ]
        e = zeros(nsdfb3d_nsubbands(fb))
        nsdfb3d_decompose(v, fb) do idx, band
            e[idx] = sum(abs2, band)
            return nothing
        end
        @test argmax(e) == predict(k)
        # The winning wedge must dominate, not merely edge out its neighbours.
        @test e[argmax(e)] / sum(e) > 0.3
    end
end

@testitem "nsdfb3d — streaming callback matches the materialised subbands" tags = [:directional] begin
    using Random
    Random.seed!(81)
    x = randn(16, 16, 16)
    fb = make_nsdfb3d(size(x), Float64; levels = 2, order = 6)
    ref = nsdfb3d_decompose(x, fb)
    seen = Int[]
    nsdfb3d_decompose(x, fb) do idx, band
        push!(seen, idx)
        @test maximum(abs, band .- ref[idx]) == 0.0
        return nothing
    end
    @test seen == collect(1:nsdfb3d_nsubbands(fb))
end

@testitem "nsdfb3d — in-place decomposition" tags = [:directional] begin
    using Random
    Random.seed!(82)
    x = randn(16, 16, 16)
    fb = make_nsdfb3d(size(x), Float64; levels = 2, order = 6)
    dst = [similar(x) for _ in 1:nsdfb3d_nsubbands(fb)]
    nsdfb3d_decompose!(dst, x, fb)
    ref = nsdfb3d_decompose(x, fb)
    @test all(i -> maximum(abs, dst[i] .- ref[i]) == 0.0, eachindex(ref))
    @test_throws DimensionMismatch nsdfb3d_decompose!(dst[1:2], x, fb)
    @test_throws DimensionMismatch nsdfb3d_reconstruct!(similar(x), dst[1:2], fb)
end

@testitem "nsdfb3d — exact circular-shift equivariance" tags = [:directional] begin
    using Random
    Random.seed!(83)
    x = randn(16, 16, 16)
    fb = make_nsdfb3d(size(x), Float64; levels = 2, order = 6)
    s = (5, -3, 2)
    a = nsdfb3d_decompose(x, fb)
    b = nsdfb3d_decompose(circshift(x, s), fb)
    for i in eachindex(a)
        @test maximum(abs, b[i] .- circshift(a[i], s)) < 1.0e-10
    end
end

@testitem "nsdfb3d — complex equals real + i·imag bit-for-bit" tags = [:directional] begin
    using Random
    Random.seed!(84)
    xr = randn(16, 16, 16); xi = randn(16, 16, 16)
    fb = make_nsdfb3d(size(xr), ComplexF64; levels = 2, order = 6)
    sc = nsdfb3d_decompose(complex.(xr, xi), fb)
    sr = nsdfb3d_decompose(xr, fb)
    si = nsdfb3d_decompose(xi, fb)
    for i in eachindex(sc)
        @test maximum(abs, sc[i] .- complex.(sr[i], si[i])) == 0.0
    end
    @test maximum(abs, nsdfb3d_reconstruct(sc, fb) .- complex.(xr, xi)) < 1.0e-12
end

@testitem "nsdfb3d — Float32" tags = [:directional] begin
    using Random
    Random.seed!(85)
    x = randn(Float32, 16, 16, 16)
    fb = make_nsdfb3d(size(x), Float32; levels = 2, order = 6)
    s = nsdfb3d_decompose(x, fb)
    @test all(b -> eltype(b) == Float32, s)
    @test maximum(abs, nsdfb3d_reconstruct(s, fb) .- x) < 1.0f-4
end

@testitem "nsdfb3d — dimension validation" tags = [:directional] begin
    fb = make_nsdfb3d((8, 8, 8), Float64; levels = 1, order = 4)
    @test_throws DimensionMismatch nsdfb3d_decompose(randn(8, 8, 4), fb)
    @test_throws DimensionMismatch nsdfb3d_decompose((i, b) -> nothing, randn(4, 8, 8), fb)
end
