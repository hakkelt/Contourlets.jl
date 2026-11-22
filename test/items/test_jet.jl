@testitem "@test_call Contourlets public API" tags = [:jet] begin
    using MultiscaleFilterBanks: upsample_filter
    using JET, Random
    Random.seed!(100)

    # ── Types / constructors ─────────────────────────────────────────────────
    @test_call target_modules = (Contourlets,) FilterPair([1.0, 2.0], [1.0, 2.0])
    @test_call target_modules = (Contourlets,) QuincunxFilterPair([0.25, 0.25])
    @test_call target_modules = (Contourlets,) CT(scales = 2, direction_levels = [2, 1])
    @test_call target_modules = (Contourlets,) TICT(scales = 2, direction_levels = [2, 1])
    @test_call target_modules = (Contourlets,) parabolic_levels(4)

    # ── Filters ──────────────────────────────────────────────────────────────
    @test_call target_modules = (Contourlets,) upsample_filter([1.0, 2.0, 3.0], 2)
    @test_call target_modules = (Contourlets,) Contourlets.upsample_kernel([1.0 2.0; 3.0 4.0], 2)

    # ── Primitives ───────────────────────────────────────────────────────────
    x = randn(16, 16)
    dst = similar(x)
    @test_call target_modules = (Contourlets,) Contourlets.conv2d_sep(x, [1.0, 2.0, 1.0], [1.0, 2.0, 1.0])
    @test_call target_modules = (Contourlets,) Contourlets.conv2d_sep!(dst, x, [1.0, 2.0, 1.0], [1.0, 2.0, 1.0])

    @test_call target_modules = (Contourlets,) Contourlets.rect_downsample(x)
    y = Contourlets.rect_downsample(x)
    @test_call target_modules = (Contourlets,) Contourlets.rect_upsample(y)
    dst2 = zeros(4, 4)
    @test_call target_modules = (Contourlets,) Contourlets.rect_downsample!(dst2, x)
    @test_call target_modules = (Contourlets,) Contourlets.qx_downsample(x)
    qy = Contourlets.qx_downsample(x)
    @test_call target_modules = (Contourlets,) Contourlets.qx_upsample(qy, (16, 16))

    # ── Pyramid ──────────────────────────────────────────────────────────────
    @test_call target_modules = (Contourlets,) Contourlets.lp_decompose(x, CDF97)
    c, bp = Contourlets.lp_decompose(x, CDF97)
    @test_call target_modules = (Contourlets,) Contourlets.lp_reconstruct(c, bp, CDF97)
    @test_call target_modules = (Contourlets,) Contourlets.nsp_decompose(x, CDF97, 1)
    nc, nbp = Contourlets.nsp_decompose(x, CDF97, 1)
    @test_call target_modules = (Contourlets,) Contourlets.nsp_reconstruct(nc, nbp, CDF97, 1)

    # ── Directional FB ───────────────────────────────────────────────────────
    @test_call target_modules = (Contourlets,) Contourlets.qfb_decompose(x, Q2345)
    sb0, sb1 = Contourlets.qfb_decompose(x, Q2345)
    @test_call target_modules = (Contourlets,) Contourlets.qfb_reconstruct(sb0, sb1, Q2345)
    @test_call target_modules = (Contourlets,) Contourlets.dfb_decompose(x, Q2345, 2)
    sbs = Contourlets.dfb_decompose(x, Q2345, 2)
    @test_call target_modules = (Contourlets,) Contourlets.dfb_reconstruct(sbs, Q2345)
    @test_call target_modules = (Contourlets,) Contourlets.nsdfb_decompose(x, Q2345, 2; tree_level = 1)
    nsbs = Contourlets.nsdfb_decompose(x, Q2345, 2; tree_level = 1)
    @test_call target_modules = (Contourlets,) Contourlets.nsdfb_reconstruct(nsbs, Q2345; tree_level = 1)

    # ── Transforms, allocating and with workspaces ───────────────────────────
    ts = (CT(scales = 2, direction_levels = [2, 1]), NSCT(scales = 2, direction_levels = [2, 1]), TICT(scales = 2, direction_levels = [2, 1]), STICT(scales = 2, direction_levels = [2, 1]))
    rec = similar(x)
    for t in ts
        @test_call target_modules = (Contourlets,) analyze(t, x)
        c = analyze(t, x)
        @test_call target_modules = (Contourlets,) synthesize(t, c)
        @test_call target_modules = (Contourlets,) make_workspace(t, Float64, (16, 16))
        @test_call target_modules = (Contourlets,) make_workspace(t, x)
        @test_call target_modules = (Contourlets,) similar_coefficients(t, x)
        @test_call target_modules = (Contourlets,) subband_layout(t, (16, 16))
        ws = make_workspace(t, x)
        @test_call target_modules = (Contourlets,) analyze!(c, t, x; workspace = ws)
        @test_call target_modules = (Contourlets,) synthesize!(rec, t, c; workspace = ws)
        @test_call target_modules = (Contourlets,) analyze!(c, ws, x)
        @test_call target_modules = (Contourlets,) synthesize(ws, c)
    end
end

@testitem "@test_opt Contourlets core functions" tags = [:jet] begin
    using JET, Random
    Random.seed!(101)
    x = randn(16, 16)
    dst = similar(x)

    # ── Primitives ───────────────────────────────────────────────────────────
    @test_opt target_modules = (Contourlets,) Contourlets.rect_downsample(x)
    @test_opt target_modules = (Contourlets,) Contourlets.conv2d_sep(x, [1.0, 2.0, 1.0], [1.0, 2.0, 1.0])

    # ── Pyramid ──────────────────────────────────────────────────────────────
    @test_opt target_modules = (Contourlets,) Contourlets.lp_decompose(x, CDF97)
    c, bp = Contourlets.lp_decompose(x, CDF97)
    @test_opt target_modules = (Contourlets,) Contourlets.lp_reconstruct(c, bp, CDF97)
    @test_opt target_modules = (Contourlets,) Contourlets.nsp_decompose(x, CDF97, 1)

    # ── Directional FB ───────────────────────────────────────────────────────
    @test_opt target_modules = (Contourlets,) Contourlets.qfb_decompose(x, Q2345)
    sb0, sb1 = Contourlets.qfb_decompose(x, Q2345)
    @test_opt target_modules = (Contourlets,) Contourlets.qfb_reconstruct(sb0, sb1, Q2345)
    @test_opt target_modules = (Contourlets,) Contourlets.dfb_decompose(x, Q2345, 2)
    sbs = Contourlets.dfb_decompose(x, Q2345, 2)
    @test_opt target_modules = (Contourlets,) Contourlets.dfb_reconstruct(sbs, Q2345)

    # ── Transforms ───────────────────────────────────────────────────────────
    # The workspace paths are the hot loops: no dynamic dispatch.
    ts = (CT(scales = 2, direction_levels = [2, 1]), NSCT(scales = 2, direction_levels = [2, 1]), TICT(scales = 2, direction_levels = [2, 1]), STICT(scales = 2, direction_levels = [2, 1]))
    rec = similar(x)
    for t in ts
        @test_opt target_modules = (Contourlets,) analyze(t, x)
        c = analyze(t, x)
        @test_opt target_modules = (Contourlets,) synthesize(t, c)
        ws = make_workspace(t, x)
        @test_opt target_modules = (Contourlets,) analyze!(c, t, x; workspace = ws)
        @test_opt target_modules = (Contourlets,) synthesize!(rec, t, c; workspace = ws)
    end
end

@testitem "@test_call Contourlets in-place and threading API" tags = [:jet] begin
    using JET, Random
    Random.seed!(102)
    M = (Contourlets,)
    x = randn(16, 16)

    # ── Threading policies ───────────────────────────────────────────────────
    @test_call target_modules = M FilterPair(CDF97.h, CDF97.g)
    @test_call target_modules = M analyze(CT(scales = 2, direction_levels = [2, 1]), x; threaded = false)
    @test_call target_modules = M analyze(NSCT(scales = 2, direction_levels = [2, 1]), x; threaded = true)

    # ── Primitives (in place) ────────────────────────────────────────────────
    y = Contourlets.rect_downsample(x)
    @test_call target_modules = M Contourlets.rect_upsample!(similar(x), y)
    qy = Contourlets.qx_downsample(x)
    @test_call target_modules = M Contourlets.qx_downsample!(similar(qy), x)
    @test_call target_modules = M Contourlets.qx_upsample!(similar(x), qy)

    # ── Pyramids (in place) ──────────────────────────────────────────────────
    c, bp = Contourlets.lp_decompose(x, CDF97)
    @test_call target_modules = M Contourlets.lp_decompose!(similar(c), similar(bp), x, CDF97)
    @test_call target_modules = M Contourlets.lp_reconstruct!(similar(x), c, bp, CDF97)
    nc, nbp = Contourlets.nsp_decompose(x, CDF97, 1)
    @test_call target_modules = M Contourlets.nsp_decompose!(similar(nc), similar(nbp), x, CDF97, 1)
    @test_call target_modules = M Contourlets.nsp_reconstruct!(similar(x), nc, nbp, CDF97, 1)

    # ── Directional filter banks (in place) ──────────────────────────────────
    sb0, sb1 = Contourlets.qfb_decompose(x, Q2345)
    @test_call target_modules = M Contourlets.qfb_decompose!(similar(sb0), similar(sb1), x, Q2345)
    @test_call target_modules = M Contourlets.qfb_reconstruct!(similar(x), sb0, sb1, Q2345)
    @test_call target_modules = M Contourlets.dfb_subband_sizes((16, 16), 2)
    sbs = Contourlets.dfb_decompose(x, Q2345, 2)
    @test_call target_modules = M Contourlets.dfb_decompose!([similar(s) for s in sbs], x, Q2345, 2)
    @test_call target_modules = M Contourlets.dfb_reconstruct!(similar(x), sbs, Q2345)

    # ── Workspaces and coefficients, threaded and not ───────────────────────
    rec = similar(x)
    for t in (NSCT(scales = 2, direction_levels = [2, 1]), TICT(scales = 2, direction_levels = [2, 1]))
        ws = make_workspace(t, x)
        c = similar_coefficients(t, x)
        @test c isa Array{Float64, 3}
        for threaded in (false, true)
            @test_call target_modules = M analyze!(c, t, x; workspace = ws, threaded)
            @test_call target_modules = M synthesize!(rec, t, c; workspace = ws, threaded)
        end
    end
    @test make_workspace(CT(scales = 2, direction_levels = [2, 1]), x) isa CTWorkspace
    @test make_workspace(NSCT(scales = 2, direction_levels = [2, 1]), x) isa NSCTWorkspace
    @test make_workspace(TICT(scales = 2, direction_levels = [2, 1]), x) isa TICTWorkspace
    @test make_workspace(STICT(scales = 2, direction_levels = [2, 1]), x) isa STICTWorkspace
    @test similar_coefficients(CT(scales = 2, direction_levels = [2, 1]), x) isa SubbandVector{Float64}
    @test similar_coefficients(STICT(scales = 2, direction_levels = [2, 1]), x) isa SubbandVector{Float64}
end

@testitem "@test_opt Contourlets remaining API" tags = [:jet] begin
    using MultiscaleFilterBanks: upsample_filter
    using JET, Random
    Random.seed!(103)
    M = (Contourlets,)
    x = randn(16, 16)
    h = [1.0, 2.0, 1.0]
    dst = similar(x)

    # ── Types / constructors / filters ───────────────────────────────────────
    @test_opt target_modules = M FilterPair([1.0, 2.0], [1.0, 2.0])
    @test_opt target_modules = M QuincunxFilterPair([0.25, 0.25])
    @test_opt target_modules = M NSCT(scales = 2, direction_levels = [2, 1])
    @test_opt target_modules = M parabolic_levels(4)
    @test_opt target_modules = M upsample_filter([1.0, 2.0, 3.0], 2)
    @test_opt target_modules = M Contourlets.upsample_kernel([1.0 2.0; 3.0 4.0], 2)

    # ── Primitives ───────────────────────────────────────────────────────────
    @test_opt target_modules = M Contourlets.conv2d_sep!(dst, x, h, h)
    y = Contourlets.rect_downsample(x)
    qy = Contourlets.qx_downsample(x)
    @test_opt target_modules = M Contourlets.rect_downsample!(similar(y), x)
    @test_opt target_modules = M Contourlets.rect_upsample(y)
    @test_opt target_modules = M Contourlets.rect_upsample!(similar(x), y)
    @test_opt target_modules = M Contourlets.qx_downsample(x)
    @test_opt target_modules = M Contourlets.qx_downsample!(similar(qy), x)
    @test_opt target_modules = M Contourlets.qx_upsample(qy, (16, 16))
    @test_opt target_modules = M Contourlets.qx_upsample!(similar(x), qy)

    # ── Pyramids ─────────────────────────────────────────────────────────────
    c, bp = Contourlets.lp_decompose(x, CDF97)
    @test_opt target_modules = M Contourlets.lp_decompose!(similar(c), similar(bp), x, CDF97)
    @test_opt target_modules = M Contourlets.lp_reconstruct!(similar(x), c, bp, CDF97)
    nc, nbp = Contourlets.nsp_decompose(x, CDF97, 1)
    @test_opt target_modules = M Contourlets.nsp_decompose!(similar(nc), similar(nbp), x, CDF97, 1)
    @test_opt target_modules = M Contourlets.nsp_reconstruct(nc, nbp, CDF97, 1)
    @test_opt target_modules = M Contourlets.nsp_reconstruct!(similar(x), nc, nbp, CDF97, 1)

    # ── Directional filter banks ─────────────────────────────────────────────
    sb0, sb1 = Contourlets.qfb_decompose(x, Q2345)
    @test_opt target_modules = M Contourlets.qfb_decompose!(similar(sb0), similar(sb1), x, Q2345)
    @test_opt target_modules = M Contourlets.qfb_reconstruct!(similar(x), sb0, sb1, Q2345)
    @test_opt target_modules = M Contourlets.dfb_subband_sizes((16, 16), 2)
    sbs = Contourlets.dfb_decompose(x, Q2345, 2)
    @test_opt target_modules = M Contourlets.dfb_decompose!([similar(s) for s in sbs], x, Q2345, 2)
    @test_opt target_modules = M Contourlets.dfb_reconstruct!(similar(x), sbs, Q2345)
    @test_opt target_modules = M Contourlets.nsdfb_decompose(x, Q2345, 2; tree_level = 1)
    nsbs = Contourlets.nsdfb_decompose(x, Q2345, 2; tree_level = 1)
    @test_opt target_modules = M Contourlets.nsdfb_reconstruct(nsbs, Q2345; tree_level = 1)

    # ── Workspaces and coefficients ──────────────────────────────────────────
    ts = (CT(scales = 2, direction_levels = [2, 1]), NSCT(scales = 2, direction_levels = [2, 1]), TICT(scales = 2, direction_levels = [2, 1]), STICT(scales = 2, direction_levels = [2, 1]))
    for t in ts
        @test_opt target_modules = M make_workspace(t, Float64, (16, 16))
        @test_opt target_modules = M make_workspace(t, x)
        @test_opt target_modules = M similar_coefficients(t, x)
        ws = make_workspace(t, x)
        c = similar_coefficients(t, x)
        @test_opt target_modules = M analyze!(c, ws, x)
        @test_opt target_modules = M synthesize!(dst, ws, c)
        @test_opt target_modules = M synthesize(t, c; workspace = ws)
    end
end
