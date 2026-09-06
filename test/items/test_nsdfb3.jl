@testitem "nsdfb3 — subband counting and index round trip" tags = [:directional] begin
    for L in 0:3
        fb = make_nsdfb3((8, 8, 8), Float64; levels = L, order = 4)
        @test nsdfb3_nsubbands(fb) == 3 * 2^L
        na, nb = 1 << fb.levels[1], 1 << fb.levels[2]
        @test na * nb == 2^L
        seen = Int[]
        for i in 1:3, p in 1:na, q in 1:nb
            idx = nsdfb3_subband_index(fb, i, p, q)
            push!(seen, idx)
            @test Contourlets._subband_triple(fb, idx) == (i, p, q)
        end
        @test sort(seen) == collect(1:nsdfb3_nsubbands(fb))
    end
    fb = make_nsdfb3((8, 8, 8), Float64; levels = 2, order = 4)
    @test_throws ArgumentError nsdfb3_subband_index(fb, 4, 1, 1)
    @test_throws ArgumentError nsdfb3_subband_index(fb, 1, 99, 1)
    @test_throws ArgumentError Contourlets._subband_triple(fb, 0)
    @test_throws ArgumentError make_nsdfb3((8, 8, 8), Float64; levels = -1)
end

@testitem "nsdfb3 — masks are a partition of unity" tags = [:directional] begin
    for L in 0:3
        fb = make_nsdfb3((10, 12, 8), Float64; levels = L, order = 6)
        total = sum(nsdfb3_mask(fb, i) for i in 1:nsdfb3_nsubbands(fb))
        @test maximum(abs, total .- 1) < 1.0e-13
    end
end

@testitem "nsdfb3 — wedges of one hourglass sum to that hourglass mask" tags = [:directional] begin
    fb = make_nsdfb3((12, 12, 12), Float64; levels = 3, order = 6)
    na, nb = 1 << fb.levels[1], 1 << fb.levels[2]
    for i in 1:3
        acc = sum(
            nsdfb3_mask(fb, nsdfb3_subband_index(fb, i, p, q))
                for p in 1:na, q in 1:nb
        )
        @test maximum(abs, acc .- hourglass3_mask(fb.fb, i)) < 1.0e-13
    end
end

@testitem "nsdfb3 — perfect reconstruction" tags = [:directional] begin
    using Random
    Random.seed!(80)
    for dims in ((16, 16, 16), (12, 20, 8))
        x = randn(dims...)
        for L in 0:3, tr in (0.0, 0.5, 1.0)
            fb = make_nsdfb3(dims, Float64; levels = L, order = 6, transition = tr)
            s = nsdfb3_decompose(x, fb)
            @test length(s) == nsdfb3_nsubbands(fb)
            @test maximum(abs, nsdfb3_reconstruct(s, fb) .- x) < 1.0e-12
        end
    end
end

@testitem "nsdfb3 — plane waves land in the predicted wedge" tags = [:directional] begin
    n = 32
    fb = make_nsdfb3((n, n, n), Float64; levels = 4, order = 12, transition = 0.5)
    la, lb = fb.levels
    transverse(i) = i == 1 ? (2, 3) : (i == 2 ? (1, 3) : (1, 2))
    function predict(k)
        i = argmax(abs.(k))
        a, b = transverse(i)
        p = clamp(floor(Int, (k[a] / k[i] + 1) / (2 / (1 << la))) + 1, 1, 1 << la)
        q = clamp(floor(Int, (k[b] / k[i] + 1) / (2 / (1 << lb))) + 1, 1, 1 << lb)
        return nsdfb3_subband_index(fb, i, p, q)
    end
    # Frequencies chosen away from wedge and hourglass boundaries: no two
    # magnitudes equal, and neither ratio lands on a wedge edge.
    for k in ([2, 5, 9], [5, 9, -12], [-2, 3, -13], [-13, -9, 7], [9, 7, -12], [3, 11, -7])
        v = [
            cospi(2 * (k[1] * (I[1] - 1) + k[2] * (I[2] - 1) + k[3] * (I[3] - 1)) / n)
                for I in CartesianIndices((n, n, n))
        ]
        e = zeros(nsdfb3_nsubbands(fb))
        nsdfb3_decompose(v, fb) do idx, band
            e[idx] = sum(abs2, band)
            return nothing
        end
        @test argmax(e) == predict(k)
        # The winning wedge must dominate, not merely edge out its neighbours.
        @test e[argmax(e)] / sum(e) > 0.3
    end
end

@testitem "nsdfb3 — streaming callback matches the materialised subbands" tags = [:directional] begin
    using Random
    Random.seed!(81)
    x = randn(16, 16, 16)
    fb = make_nsdfb3(size(x), Float64; levels = 2, order = 6)
    ref = nsdfb3_decompose(x, fb)
    seen = Int[]
    nsdfb3_decompose(x, fb) do idx, band
        push!(seen, idx)
        @test maximum(abs, band .- ref[idx]) == 0.0
        return nothing
    end
    @test seen == collect(1:nsdfb3_nsubbands(fb))
end

@testitem "nsdfb3 — in-place decomposition" tags = [:directional] begin
    using Random
    Random.seed!(82)
    x = randn(16, 16, 16)
    fb = make_nsdfb3(size(x), Float64; levels = 2, order = 6)
    dst = [similar(x) for _ in 1:nsdfb3_nsubbands(fb)]
    nsdfb3_decompose!(dst, x, fb)
    ref = nsdfb3_decompose(x, fb)
    @test all(i -> maximum(abs, dst[i] .- ref[i]) == 0.0, eachindex(ref))
    @test_throws DimensionMismatch nsdfb3_decompose!(dst[1:2], x, fb)
    @test_throws DimensionMismatch nsdfb3_reconstruct!(similar(x), dst[1:2], fb)
end

@testitem "nsdfb3 — exact circular-shift equivariance" tags = [:directional] begin
    using Random
    Random.seed!(83)
    x = randn(16, 16, 16)
    fb = make_nsdfb3(size(x), Float64; levels = 2, order = 6)
    s = (5, -3, 2)
    a = nsdfb3_decompose(x, fb)
    b = nsdfb3_decompose(circshift(x, s), fb)
    for i in eachindex(a)
        @test maximum(abs, b[i] .- circshift(a[i], s)) < 1.0e-10
    end
end

@testitem "nsdfb3 — complex equals real + i·imag bit-for-bit" tags = [:directional] begin
    using Random
    Random.seed!(84)
    xr = randn(16, 16, 16); xi = randn(16, 16, 16)
    fb = make_nsdfb3(size(xr), ComplexF64; levels = 2, order = 6)
    sc = nsdfb3_decompose(complex.(xr, xi), fb)
    sr = nsdfb3_decompose(xr, fb)
    si = nsdfb3_decompose(xi, fb)
    for i in eachindex(sc)
        @test maximum(abs, sc[i] .- complex.(sr[i], si[i])) == 0.0
    end
    @test maximum(abs, nsdfb3_reconstruct(sc, fb) .- complex.(xr, xi)) < 1.0e-12
end

@testitem "nsdfb3 — Float32" tags = [:directional] begin
    using Random
    Random.seed!(85)
    x = randn(Float32, 16, 16, 16)
    fb = make_nsdfb3(size(x), Float32; levels = 2, order = 6)
    s = nsdfb3_decompose(x, fb)
    @test all(b -> eltype(b) == Float32, s)
    @test maximum(abs, nsdfb3_reconstruct(s, fb) .- x) < 1.0f-4
end

@testitem "nsdfb3 — dimension validation" tags = [:directional] begin
    fb = make_nsdfb3((8, 8, 8), Float64; levels = 1, order = 4)
    @test_throws DimensionMismatch nsdfb3_decompose(randn(8, 8, 4), fb)
    @test_throws DimensionMismatch nsdfb3_decompose((i, b) -> nothing, randn(4, 8, 8), fb)
end
