@testitem "Generic axes: building blocks" tags = [:quality] begin
    using Contourlets: lp_decompose, lp_decompose!, lp_reconstruct, lp_reconstruct!,
        nsp_decompose, nsp_decompose!, nsp_reconstruct, nsp_reconstruct!,
        qfb_decompose, qfb_decompose!, qfb_reconstruct, qfb_reconstruct!,
        dfb_decompose, dfb_decompose!, dfb_reconstruct, dfb_reconstruct!, dfb_subband_sizes,
        nsdfb_decompose, nsdfb_reconstruct, upsample_kernel
    using OffsetArrays, Random

    msg = "offset arrays are not supported"
    rng = Xoshiro(3)
    x = randn(rng, 16, 16)
    big = randn(rng, 20, 20)
    big[3:18, 2:17] .= x
    xv = view(big, 3:18, 2:17)        # a view is indexed from 1
    xo = OffsetArray(x, 0:15, 0:15)
    cs, qs = Contourlets.CDF97, Contourlets.Q2345
    offset(A) = OffsetArray(A, ntuple(_ -> -1, ndims(A)))
    bigv(A) = (B = similar(A, size(A) .+ 2); B .= 0; copyto!(view(B, 2:(size(A, 1) + 1), 2:(size(A, 2) + 1)), A); view(B, 2:(size(A, 1) + 1), 2:(size(A, 2) + 1)))

    @testset "conv2d_sep, sampling" begin
        h = [0.25, 0.5, 0.25]
        for boundary in (:symmetric, :periodic, :zero)
            @test Contourlets.conv2d_sep(xv, h, h; boundary) == Contourlets.conv2d_sep(x, h, h; boundary)
            @test_throws msg Contourlets.conv2d_sep(xo, h, h; boundary)
            @test_throws msg Contourlets.conv2d_sep(x, offset(h), h; boundary)
            @test_throws msg Contourlets.conv2d_sep!(similar(xo), x, h, h; boundary)
            @test_throws msg Contourlets.conv2d_sep!(similar(x), x, h, h; boundary, tmp = similar(xo))
        end
        @test Contourlets.rect_downsample(xv) == Contourlets.rect_downsample(x)
        @test_throws msg Contourlets.rect_downsample(xo)
        @test_throws msg Contourlets.rect_upsample(xo)
        @test Contourlets.qx_downsample(xv) == Contourlets.qx_downsample(x)
        @test_throws msg Contourlets.qx_downsample(xo)
        @test_throws msg Contourlets.qx_downsample!(zeros(16, 8), xo)
        @test_throws msg Contourlets.qx_upsample!(xo, zeros(16, 8))
        @test_throws msg Contourlets.qx_upsample!(zeros(16, 16), offset(zeros(16, 8)))
    end

    @testset "upsample_kernel" begin
        K = [1.0 2.0; 3.0 4.0]
        @test upsample_kernel(view(K, :, :), 2) == upsample_kernel(K, 2) == [1.0 0.0 2.0; 0.0 0.0 0.0; 3.0 0.0 4.0]
        @test upsample_kernel(view(K, :, :), 1) == K
        @test_throws msg upsample_kernel(offset(K), 2)
        @test_throws msg upsample_kernel(offset(K), 1)
    end

    @testset "Laplacian and nonsubsampled pyramids" begin
        c, b = lp_decompose(x, cs)
        @test lp_decompose(xv, cs) == (c, b)
        @test lp_reconstruct(view(c, :, :), view(b, :, :), cs) == lp_reconstruct(c, b, cs)
        @test_throws msg lp_decompose(xo, cs)
        @test_throws msg lp_reconstruct(offset(c), b, cs)
        @test_throws msg lp_reconstruct(c, offset(b), cs)
        cv, bv = bigv(zero(c)), bigv(zero(b))
        lp_decompose!(cv, bv, xv, cs)
        @test (cv, bv) == (c, b)
        @test_throws msg lp_decompose!(c, b, xo, cs)
        @test_throws msg lp_decompose!(offset(c), b, x, cs)
        @test_throws msg lp_decompose!(c, offset(b), x, cs)
        @test_throws msg lp_decompose!(c, b, x, cs; tmp = offset(b))
        out = bigv(zero(x))
        lp_reconstruct!(out, bigv(c), bigv(b), cs)
        @test out == lp_reconstruct(c, b, cs)
        @test_throws msg lp_reconstruct!(xo, c, b, cs)
        @test_throws msg lp_reconstruct!(out, offset(c), b, cs)
        @test_throws msg lp_reconstruct!(out, c, offset(b), cs)

        for level in 1:2
            c, b = nsp_decompose(x, cs, level)
            @test nsp_decompose(xv, cs, level) == (c, b)
            @test nsp_reconstruct(view(c, :, :), view(b, :, :), cs, level) == nsp_reconstruct(c, b, cs, level)
            @test_throws msg nsp_decompose(xo, cs, level)
            @test_throws msg nsp_reconstruct(offset(c), b, cs, level)
            @test_throws msg nsp_reconstruct(c, offset(b), cs, level)
            cv, bv = bigv(zero(c)), bigv(zero(b))
            nsp_decompose!(cv, bv, xv, cs, level)
            @test (cv, bv) == (c, b)
            @test_throws msg nsp_decompose!(c, b, xo, cs, level)
            @test_throws msg nsp_decompose!(offset(c), b, x, cs, level)
            @test_throws msg nsp_decompose!(c, offset(b), x, cs, level)
            out = bigv(zero(x))
            nsp_reconstruct!(out, bigv(c), bigv(b), cs, level)
            @test out == nsp_reconstruct(c, b, cs, level)
            @test_throws msg nsp_reconstruct!(xo, c, b, cs, level)
            @test_throws msg nsp_reconstruct!(out, offset(c), b, cs, level)
            @test_throws msg nsp_reconstruct!(out, c, offset(b), cs, level)
        end
    end

    @testset "Quincunx filter bank" begin
        for dir in (:col, :row)
            s0, s1 = qfb_decompose(x, qs; dir)
            @test qfb_decompose(xv, qs; dir) == (s0, s1)
            @test qfb_reconstruct(view(s0, :, :), view(s1, :, :), qs; dir) == qfb_reconstruct(s0, s1, qs; dir)
            @test_throws msg qfb_decompose(xo, qs; dir)
            @test_throws msg qfb_reconstruct(offset(s0), s1, qs; dir)
            @test_throws msg qfb_reconstruct(s0, offset(s1), qs; dir)
            t0, t1 = bigv(zero(s0)), bigv(zero(s1))
            qfb_decompose!(t0, t1, xv, qs; dir)
            @test (t0, t1) == (s0, s1)
            @test_throws msg qfb_decompose!(t0, t1, xo, qs; dir)
            @test_throws msg qfb_decompose!(offset(t0), t1, x, qs; dir)
            @test_throws msg qfb_decompose!(t0, offset(t1), x, qs; dir)
            out = bigv(zero(x))
            qfb_reconstruct!(out, bigv(s0), bigv(s1), qs; dir)
            @test out == qfb_reconstruct(s0, s1, qs; dir)
            @test_throws msg qfb_reconstruct!(xo, s0, s1, qs; dir)
            @test_throws msg qfb_reconstruct!(out, offset(s0), s1, qs; dir)
            @test_throws msg qfb_reconstruct!(out, s0, offset(s1), qs; dir)
        end
    end

    @testset "Directional filter banks (l = $l)" for l in 0:2
        sb = dfb_decompose(x, qs, l)
        @test dfb_decompose(xv, qs, l) == sb
        @test dfb_reconstruct(map(s -> view(s, :, :), sb), qs) == dfb_reconstruct(sb, qs)
        @test_throws msg dfb_decompose(xo, qs, l)
        @test_throws msg dfb_reconstruct(offset.(sb), qs)
        @test_throws msg dfb_reconstruct(OffsetArray(sb, 0:(length(sb) - 1)), qs)
        out = [bigv(zero(s)) for s in sb]
        dfb_decompose!(out, xv, qs, l)
        @test out == sb
        @test_throws msg dfb_decompose!(out, xo, qs, l)
        @test_throws msg dfb_decompose!(offset.(out), x, qs, l)
        @test_throws msg dfb_decompose!(OffsetArray(out, 0:(length(out) - 1)), x, qs, l)
        y = bigv(zero(x))
        dfb_reconstruct!(y, map(bigv, sb), qs)
        @test y == dfb_reconstruct(sb, qs)
        @test_throws msg dfb_reconstruct!(xo, sb, qs)
        @test_throws msg dfb_reconstruct!(y, offset.(sb), qs)
        @test_throws msg dfb_reconstruct!(y, OffsetArray(sb, 0:(length(sb) - 1)), qs)

        ns = nsdfb_decompose(x, qs, l; tree_level = 2)
        @test nsdfb_decompose(xv, qs, l; tree_level = 2) == ns
        @test nsdfb_reconstruct(map(s -> view(s, :, :), ns), qs; tree_level = 2) == nsdfb_reconstruct(ns, qs; tree_level = 2)
        @test_throws msg nsdfb_decompose(xo, qs, l)
        @test_throws msg nsdfb_reconstruct(offset.(ns), qs)
        @test_throws msg nsdfb_reconstruct(OffsetArray(ns, 0:(length(ns) - 1)), qs)
    end
end

@testitem "Generic axes: transforms" tags = [:quality] begin
    using Contourlets: CT, NSCT, TICT, STICT, analyze, analyze!, synthesize, synthesize!, make_workspace,
        similar_coefficients, nsubbands, subband, SubbandVector
    using OffsetArrays, Random

    msg = "offset arrays are not supported"
    rng = Xoshiro(11)
    dims = (32, 32)
    x = randn(rng, dims...)
    big = randn(rng, 36, 36)
    big[3:34, 2:33] .= x
    xv = view(big, 3:34, 2:33)        # a view is indexed from 1
    xo = OffsetArray(x, 0:31, 0:31)

    transforms = (
        CT(scales = 2, direction_levels = [1, 2]),
        NSCT(scales = 2, direction_levels = [1, 2]),
        TICT(scales = 2, direction_levels = [1, 2]),
        STICT(scales = 2, direction_levels = [1, 2]),
    )
    coefficients(t, c) = [copy(subband(t, c, i)) for i in 1:nsubbands(t, dims)]
    # Coefficients and an output image that live inside larger arrays.
    function view_coefficients(t, c)
        if c isa SubbandVector
            return SubbandVector(copy(c.data), c.layout)
        end
        B = zeros(eltype(c), 34, 34, size(c, 3) + 1)
        V = view(B, 2:33, 2:33, 2:(size(c, 3) + 1))
        return copyto!(V, c)
    end
    offset_coefficients(c::SubbandVector) = SubbandVector(OffsetArray(c.data, 0:(length(c.data) - 1)), c.layout)
    offset_coefficients(c::AbstractArray) = OffsetArray(c, 0:(size(c, 1) - 1), 0:(size(c, 2) - 1), 0:(size(c, 3) - 1))

    @testset "$(nameof(typeof(t))), $(usews ? "workspace" : "no workspace")" for t in transforms, usews in (false, true)
        ws = usews ? make_workspace(t, x) : nothing
        kw = usews ? (; workspace = ws) : (;)
        c = analyze(t, x; kw...)
        ref = coefficients(t, c)
        xr = synthesize(t, c; kw...)

        # Views of one-based arrays give the results of the arrays they wrap.
        @test coefficients(t, analyze(t, xv; kw...)) == ref
        cv = view_coefficients(t, c)
        @test synthesize(t, cv; kw...) == xr
        yv = view(zeros(36, 36), 3:34, 2:33)
        synthesize!(yv, t, c; kw...)
        @test yv == xr
        c2 = similar_coefficients(t, x)
        analyze!(c2, t, xv; kw...)
        @test coefficients(t, c2) == ref

        # Offset axes are rejected on the image, on the coefficients and on the output.
        @test_throws msg analyze(t, xo; kw...)
        @test_throws msg analyze!(c2, t, xo; kw...)
        @test_throws msg analyze!(offset_coefficients(c2), t, x; kw...)
        @test_throws msg synthesize!(xo, t, c; kw...)
        @test_throws msg synthesize!(copy(x), t, offset_coefficients(c); kw...)
        @test_throws msg synthesize(t, offset_coefficients(c); kw...)
        @test_throws msg similar_coefficients(t, xo)
        @test_throws msg make_workspace(t, xo)
    end
end
