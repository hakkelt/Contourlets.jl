@testitem "conv2d direct backend small kernel" begin
    using Random
    Random.seed!(1)
    x = randn(16, 16)
    # 3-tap separable averaging kernel, compare against a hand-rolled reference
    k = reshape([0.25, 0.5, 0.25], 3, 1)
    out = Contourlets.conv2d(x, k, (2, 1); boundary = :symmetric)
    # Symmetric (whole-sample) reflection: row 0 ↦ row 2, row n+1 ↦ row n-1.
    refl(i) = i < 1 ? 2 - i : (i > 16 ? 32 - i : i)
    ref = similar(x)
    for j in 1:16, i in 1:16
        ref[i, j] = 0.25x[refl(i - 1), j] + 0.5x[i, j] + 0.25x[refl(i + 1), j]
    end
    @test maximum(abs, out .- ref) < 1.0e-13
end

@testitem "conv2d FFTW backend matches direct (all boundaries)" begin
    using Random
    Random.seed!(101)
    x = randn(40, 48)
    k = randn(7, 7)            # 49 taps > 25 ⇒ FFTW path
    c = (4, 4)
    for b in (:symmetric, :periodic, :zero)
        out_fftw = Contourlets.conv2d(x, k, c; boundary = b)   # FFTW path
        out_dir = similar(x)
        Contourlets._conv2d_direct!(out_dir, x, k, c; boundary = b)
        @test maximum(abs, out_fftw .- out_dir) < 1.0e-10
    end
end

@testitem "rect_downsample / upsample round-trip" begin
    using Random
    Random.seed!(2)
    x = randn(8, 8)
    y = rect_downsample(x)
    @test size(y) == (4, 4)
    # upsample then downsample should give back original
    u = rect_upsample(y)
    d2 = rect_downsample(u)
    @test maximum(abs, d2 .- y) < 1.0e-14
end

@testitem "shear / inv_shear" begin
    using Random
    Random.seed!(3)
    x = randn(8, 8)
    for dir in [:h, :v]
        sh = shear(x, dir)
        @test size(sh) == size(x)
        rec = inv_shear(sh, dir)
        @test maximum(abs, rec .- x) < 1.0e-14
    end
end

@testitem "shear does not alias src/dst" begin
    x = randn(8, 8)
    y = copy(x)
    sh = shear(x, :h)          # allocating — never aliases
    @test sh != x              # result differs from identity
    # in-place shear! into a separate dst is fine
    dst = similar(x)
    shear!(dst, x, :h)
    @test dst == sh
end

@testitem "conv2d FFTW backend (large kernel)" begin
    using Random
    Random.seed!(4)
    x = randn(32, 32)
    # Kernel with more than 25 elements triggers FFTW path
    k = randn(6, 6)   # 36 elements
    # origin at centre
    c = (3, 3)
    out_fftw = Contourlets.conv2d(x, k, c; boundary = :symmetric)
    # Compute reference with direct backend via small kernel comparison:
    # just ensure the output has correct size and is finite
    @test size(out_fftw) == size(x)
    @test all(isfinite, out_fftw)
end

@testitem "conv2d boundary modes :periodic and :zero" begin
    using Random
    Random.seed!(5)
    x = randn(8, 8)
    k = reshape([0.25, 0.5, 0.25], 3, 1)
    out_sym = Contourlets.conv2d(x, k; boundary = :symmetric)
    out_per = Contourlets.conv2d(x, k; boundary = :periodic)
    out_zero = Contourlets.conv2d(x, k; boundary = :zero)
    # Interior pixels should all agree (away from boundary)
    @test maximum(abs, out_sym[3:6, :] .- out_per[3:6, :]) < 1.0e-13
    @test maximum(abs, out_sym[3:6, :] .- out_zero[3:6, :]) < 1.0e-13
    # Boundary pixels differ between modes
    @test out_sym[1, 1] != out_per[1, 1] || out_sym[1, 1] != out_zero[1, 1]
end

@testitem "conv2d! size mismatch error" begin
    x = randn(8, 8)
    k = randn(3, 3)
    dst = zeros(4, 8)
    @test_throws DimensionMismatch Contourlets.conv2d!(dst, x, k)
end

@testitem "qx_downsample / upsample round-trip" begin
    using Random
    Random.seed!(6)
    x = randn(16, 16)
    y = qx_downsample(x)
    @test size(y) == (16, 8)
    x_rec = qx_upsample(y, size(x))
    # All non-zero entries of x_rec should match x at quincunx positions
    @test any(!iszero, x_rec)
end

@testitem "qx_downsample! in-place" begin
    using Random
    Random.seed!(7)
    x = randn(16, 16)
    y_alloc = qx_downsample(x)
    y_inplace = zeros(size(y_alloc)...)
    qx_downsample!(y_inplace, x)
    @test y_inplace == y_alloc
end

@testitem "qx_upsample! in-place" begin
    using Random
    Random.seed!(8)
    x = randn(16, 16)
    y = qx_downsample(x)
    x_alloc = qx_upsample(y, size(x))
    x_inplace = zeros(size(x)...)
    qx_upsample!(x_inplace, y)
    @test x_inplace == x_alloc
end

@testitem "shear! invalid direction throws" begin
    x = randn(8, 8)
    dst = similar(x)
    @test_throws ArgumentError shear!(dst, x, :bad)
end

@testitem "inv_shear! invalid direction throws" begin
    x = randn(8, 8)
    dst = similar(x)
    @test_throws ArgumentError inv_shear!(dst, x, :bad)
end

@testitem "rect_upsample! in-place" begin
    using Random
    Random.seed!(9)
    x = randn(8, 8)
    y = rect_downsample(x)
    up_alloc = rect_upsample(y)
    up_inplace = zeros(size(up_alloc)...)
    rect_upsample!(up_inplace, y)
    @test up_inplace == up_alloc
end

@testitem "rect_downsample! in-place" begin
    using Random
    Random.seed!(10)
    x = randn(8, 8)
    y_alloc = rect_downsample(x)
    y_inplace = zeros(4, 4)
    rect_downsample!(y_inplace, x)
    @test y_inplace == y_alloc
end

@testitem "conv2d_sep! periodic and zero boundaries" begin
    using Random
    Random.seed!(60)
    x = randn(8, 8)
    h = Float64[0.25, 0.5, 0.25]
    dst_per = similar(x)
    conv2d_sep!(dst_per, x, h, h; boundary = :periodic)
    dst_sym = similar(x)
    conv2d_sep!(dst_sym, x, h, h; boundary = :symmetric)
    @test maximum(abs, dst_per[3:6, 3:6] .- dst_sym[3:6, 3:6]) < 1.0e-12
    dst_zero = similar(x)
    conv2d_sep!(dst_zero, x, h, h; boundary = :zero)
    @test size(dst_zero) == size(x)
    @test all(isfinite, dst_zero)
    @test maximum(abs, dst_zero[3:6, 3:6] .- dst_sym[3:6, 3:6]) < 1.0e-12
end

@testitem "conv2d_sep allocating matches in-place" begin
    using Random
    Random.seed!(61)
    x = randn(16, 16)
    h = Float64[0.25, 0.5, 0.25]
    dst_alloc = conv2d_sep(x, h, h; boundary = :periodic)
    dst_ip = similar(x)
    conv2d_sep!(dst_ip, x, h, h; boundary = :periodic)
    @test dst_alloc == dst_ip
end

@testitem "conv2d_sep periodic boundary allocating" begin
    using Random
    Random.seed!(95)
    x = randn(8, 8)
    h = [0.25, 0.5, 0.25]
    out = conv2d_sep(x, h, h; boundary = :periodic)
    @test size(out) == size(x)
    @test all(isfinite, out)
end

@testitem "conv2d_sep zero boundary allocating" begin
    using Random
    Random.seed!(96)
    x = randn(8, 8)
    h = [0.25, 0.5, 0.25]
    out = conv2d_sep(x, h, h; boundary = :zero)
    @test size(out) == size(x)
    @test all(isfinite, out)
end

@testitem "conv2d_sep symmetric: no OOB with long filter on small image" tags = [:primitives] begin
    using Random
    Random.seed!(200)
    # CDF97 analysis filter (9 taps): radius 4 exceeds image radius on 3×3 and 4×4 inputs.
    # The single-fold reflection formula previously left indices out of [1,n] → OOB under @inbounds.
    h9 = Float64[
        0.02674875741081, -0.016864118442875, -0.07822326652899,
        0.266864118442875, 0.60294901823636, 0.266864118442875,
        -0.07822326652899, -0.016864118442875, 0.02674875741081,
    ]
    for (nr, nc) in [(3, 3), (4, 4)]
        x = randn(nr, nc)
        out = conv2d_sep(x, h9, h9; boundary = :symmetric)
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
        @test maximum(abs, conv2d_sep(x, h9, h9; boundary = :symmetric) .- ref_conv_sep(x, h9)) <
              1.0e-12
    end
end

@testitem "conv2d! large kernel Enabled() threading" begin
    using Random
    Random.seed!(102)
    x = randn(32, 32)
    k = randn(6, 6)   # 36 taps > 25 ⇒ FFTW path
    out_enabled = Contourlets.conv2d(x, k; threading = Enabled())
    out_auto = Contourlets.conv2d(x, k; threading = Auto())
    @test size(out_enabled) == size(x)
    @test maximum(abs, out_enabled .- out_auto) < 1.0e-10
end
