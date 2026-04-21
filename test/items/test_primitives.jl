@testitem "conv2d direct vs FFTW" begin
    using Contourlets, Random
    Random.seed!(1)
    x = randn(16, 16)
    k = randn(3, 3)
    c = (2, 2)
    out_d = Contourlets.conv2d(x, k, c; boundary = :symmetric)
    out_f = Contourlets.conv2d(x, k, c; boundary = :symmetric)
    @test maximum(abs, out_d .- out_f) < 1.0e-10
end

@testitem "rect_downsample / upsample round-trip" begin
    using Contourlets, Random
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
    using Contourlets, Random
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
    using Contourlets
    x = randn(8, 8)
    y = copy(x)
    sh = shear(x, :h)          # allocating — never aliases
    @test sh != x              # result differs from identity
    # in-place shear! into a separate dst is fine
    dst = similar(x)
    shear!(dst, x, :h)
    @test dst == sh
end
