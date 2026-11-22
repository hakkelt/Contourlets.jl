@testitem "rect_downsample / upsample round-trip" tags = [:primitives] begin
    using Random
    Random.seed!(2)
    x = randn(8, 8)
    y = Contourlets.rect_downsample(x)
    @test size(y) == (4, 4)
    # upsample then downsample should give back original
    u = Contourlets.rect_upsample(y)
    d2 = Contourlets.rect_downsample(u)
    @test maximum(abs, d2 .- y) < 1.0e-14
end

@testitem "qx_downsample / upsample round-trip" tags = [:primitives] begin
    using Random
    Random.seed!(6)

    # Square case
    x = randn(16, 16)
    y = Contourlets.qx_downsample(x)
    @test size(y) == (16, 8)
    x_rec = Contourlets.qx_upsample(y, size(x))
    @test size(x_rec) == size(x)
    # Retained positions must exactly match x; dropped positions must be zero.
    for i in 1:16, j in 1:16
        if isodd(i + j)   # odd-sum: dropped by downsampler
            @test x_rec[i, j] == 0
        else               # even-sum: retained
            @test x_rec[i, j] == x[i, j]
        end
    end

    # Non-square even case
    x2 = randn(8, 6)
    y2 = Contourlets.qx_downsample(x2)
    @test size(y2) == (8, 3)
    x2_rec = Contourlets.qx_upsample(y2, size(x2))
    for i in 1:8, j in 1:6
        if isodd(i + j)
            @test x2_rec[i, j] == 0
        else
            @test x2_rec[i, j] == x2[i, j]
        end
    end

    # Odd n2 must throw
    @test_throws ArgumentError Contourlets.qx_downsample(randn(8, 5))
    @test_throws ArgumentError Contourlets.qx_downsample!(zeros(8, 2), randn(8, 5))
end

@testitem "qx_downsample! in-place" tags = [:primitives] begin
    using Random
    Random.seed!(7)
    x = randn(16, 16)
    y_alloc = Contourlets.qx_downsample(x)
    y_inplace = zeros(size(y_alloc)...)
    Contourlets.qx_downsample!(y_inplace, x)
    @test y_inplace == y_alloc
end

@testitem "qx_upsample! in-place" tags = [:primitives] begin
    using Random
    Random.seed!(8)
    x = randn(16, 16)
    y = Contourlets.qx_downsample(x)
    x_alloc = Contourlets.qx_upsample(y, size(x))
    x_inplace = zeros(size(x)...)
    Contourlets.qx_upsample!(x_inplace, y)
    @test x_inplace == x_alloc
end

@testitem "rect_upsample! in-place" tags = [:primitives] begin
    using Random
    Random.seed!(9)
    x = randn(8, 8)
    y = Contourlets.rect_downsample(x)
    up_alloc = Contourlets.rect_upsample(y)
    up_inplace = zeros(size(up_alloc)...)
    Contourlets.rect_upsample!(up_inplace, y)
    @test up_inplace == up_alloc
end

@testitem "rect_downsample! in-place" tags = [:primitives] begin
    using Random
    Random.seed!(10)
    x = randn(8, 8)
    y_alloc = Contourlets.rect_downsample(x)
    y_inplace = zeros(4, 4)
    Contourlets.rect_downsample!(y_inplace, x)
    @test y_inplace == y_alloc
end

@testitem "conv2d_sep! periodic and zero boundaries" tags = [:primitives] begin
    using Random
    Random.seed!(60)
    # Reference: a separable convolution reading x through `get(i, n)`, which
    # returns the extended index or 0 for a zero sample.
    function ref_conv_sep(x, h, get)
        n1, n2 = size(x)
        c = (length(h) + 1) ÷ 2
        at(i, j) = (ii = get(i, n1); jj = get(j, n2); ii == 0 || jj == 0 ? zero(eltype(x)) : x[ii, jj])
        tmp = [sum(h[k] * at(i - (k - c), j) for k in eachindex(h)) for i in 1:n1, j in 1:n2]
        tat(i, j) = (jj = get(j, n2); jj == 0 ? zero(eltype(x)) : tmp[i, jj])
        return [sum(h[k] * tat(i, j - (k - c)) for k in eachindex(h)) for i in 1:n1, j in 1:n2]
    end
    periodic(i, n) = mod1(i, n)
    zeropad(i, n) = 1 <= i <= n ? i : 0
    h = Float64[0.1, 0.25, 0.3, 0.25, 0.1]
    for (nr, nc) in ((8, 8), (7, 5))
        x = randn(nr, nc)
        for (boundary, get) in ((:periodic, periodic), (:zero, zeropad))
            dst = similar(x)
            Contourlets.conv2d_sep!(dst, x, h, h; boundary)
            @test dst ≈ ref_conv_sep(x, h, get) atol = 1.0e-12
            @test Contourlets.conv2d_sep(x, h, h; boundary) == dst
        end
    end
end

@testitem "conv2d_sep allocating matches in-place" tags = [:primitives] begin
    using Random
    Random.seed!(61)
    x = randn(16, 16)
    h = Float64[0.25, 0.5, 0.25]
    dst_alloc = Contourlets.conv2d_sep(x, h, h; boundary = :periodic)
    dst_ip = similar(x)
    Contourlets.conv2d_sep!(dst_ip, x, h, h; boundary = :periodic)
    @test dst_alloc == dst_ip
end

@testitem "conv2d_sep symmetric: no OOB with long filter on small image" tags = [:primitives] begin
    using Random
    Random.seed!(200)
    # CDF97 analysis filter (9 taps): radius 4 exceeds image radius on 3×3 and 4×4 inputs.
    # The reflection must fold indices back into 1:n however far the filter reaches
    # past the edge: a single reflection would index out of bounds under @inbounds.
    h9 = Float64[
        0.02674875741081, -0.016864118442875, -0.07822326652899,
        0.266864118442875, 0.60294901823636, 0.266864118442875,
        -0.07822326652899, -0.016864118442875, 0.02674875741081,
    ]
    for (nr, nc) in [(3, 3), (4, 4)]
        x = randn(nr, nc)
        out = Contourlets.conv2d_sep(x, h9, h9; boundary = :symmetric)
        @test all(isfinite, out)
        @test size(out) == (nr, nc)
    end
end

@testitem "conv2d_sep symmetric: triangle-wave fold matches reference" tags = [:primitives] begin
    using Random
    Random.seed!(201)
    # Independent reference: manually fold indices with the triangle-wave formula,
    # then convolve — no dependency on the implementation under test.
    function fold_idx(i, n)
        n == 1 && return 1
        p = 2 * (n - 1)
        m = mod(i - 1, p)
        return m < n ? m + 1 : p - m + 1
    end
    function ref_conv_sep(x, h)
        n1, n2 = size(x)
        c = (length(h) + 1) ÷ 2
        tmp = similar(x)
        for j in 1:n2, i in 1:n1
            acc = zero(eltype(x))
            for k in 1:length(h)
                acc += h[k] * x[fold_idx(i - (k - c), n1), j]
            end
            tmp[i, j] = acc
        end
        out = similar(x)
        for j in 1:n2, i in 1:n1
            acc = zero(eltype(x))
            for k in 1:length(h)
                acc += h[k] * tmp[i, fold_idx(j - (k - c), n2)]
            end
            out[i, j] = acc
        end
        return out
    end
    h9 = Float64[
        0.02674875741081, -0.016864118442875, -0.07822326652899,
        0.266864118442875, 0.60294901823636, 0.266864118442875,
        -0.07822326652899, -0.016864118442875, 0.02674875741081,
    ]
    for (nr, nc) in [(3, 3), (4, 4)]
        x = randn(nr, nc)
        @test maximum(abs, Contourlets.conv2d_sep(x, h9, h9; boundary = :symmetric) .- ref_conv_sep(x, h9)) <
            1.0e-12
    end
end

@testitem "even-length filters: fast path matches generic path" tags = [:primitives] begin
    using Random
    Random.seed!(42)

    # The real fast paths split the domain into a border band and an interior
    # slab that indexes `src` unchecked.  For an even-length filter the tap loop
    # reaches one sample further back than for an odd one, so the slab's lower
    # bound is `l - c + 1`, not `c`.  Complex data takes the fully folded
    # generic path, and the transform is real-linear, so the real part of the
    # complex result is the reference for the real fast path.
    h4 = Float64[0.1, 0.4, 0.35, 0.15]
    h6 = Float64[0.05, 0.15, 0.3, 0.28, 0.15, 0.07]

    for boundary in (:symmetric, :periodic), h in (h4, h6)
        for (n1, n2) in ((8, 6), (16, 16), (5, 4))
            x = randn(n1, n2)
            fast = Contourlets.conv2d_sep(x, h, h; boundary = boundary)
            ref = real(Contourlets.conv2d_sep(complex(x), h, h; boundary = boundary))
            @test maximum(abs, fast .- ref) < 1.0e-12
        end
    end
end

@testitem "unknown boundary modes throw" tags = [:primitives] begin
    x = randn(16, 16)
    h = [0.25, 0.5, 0.25]
    msg = "boundary must be :symmetric, :periodic or :zero, got :reflect"
    @test_throws msg Contourlets.conv2d_sep(x, h, h; boundary = :reflect)
    @test_throws msg Contourlets.conv2d_sep!(similar(x), x, h, h; boundary = :reflect)
end

@testitem "rect up/downsampling checks the destination size" tags = [:primitives] begin
    x = randn(8, 8)
    @test_throws "must have size cld.(fine size, 2) = (4, 4), got (3, 4)" Contourlets.rect_downsample!(zeros(3, 4), x)
    @test_throws "must have size cld.(fine size, 2) = (5, 4), got (4, 5)" Contourlets.rect_upsample!(zeros(9, 8), randn(4, 5))
    @test_throws DimensionMismatch Contourlets.rect_upsample!(zeros(10, 8), randn(4, 4))
    # Odd fine sizes: the coarse grid is cld(n, 2), as in the Laplacian pyramid.
    y = randn(7, 9)
    c = Contourlets.rect_downsample!(zeros(4, 5), y)
    @test c == y[1:2:end, 1:2:end]
    u = Contourlets.rect_upsample!(zeros(7, 9), c)
    @test u[1:2:end, 1:2:end] == c
    @test count(!iszero, u) == count(!iszero, c)
end
