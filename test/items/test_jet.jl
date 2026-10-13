@testitem "@test_call Contourlets public API" tags = [:jet] begin
    using JET, Random
    Random.seed!(100)

    # ── Types / constructors ─────────────────────────────────────────────────
    @test_call target_modules = (Contourlets,) FilterPair([1.0, 2.0], [1.0, 2.0])
    @test_call target_modules = (Contourlets,) QuincunxFilterPair([0.25, 0.25])
    @test_call target_modules = (Contourlets,) ContourletParams(J = 2, L_array = [1, 2])
    @test_call target_modules = (Contourlets,) parabolic_levels(4)

    # ── Filters ──────────────────────────────────────────────────────────────
    @test_call target_modules = (Contourlets,) upsample_filter([1.0, 2.0, 3.0], 2)
    @test_call target_modules = (Contourlets,) upsample_kernel([1.0 2.0; 3.0 4.0], 2)
    @test_call target_modules = (Contourlets,) check_pr_condition(Q2345)
    @test_call target_modules = (Contourlets,) check_pr_condition(CDF97)

    # ── Primitives ───────────────────────────────────────────────────────────
    x = randn(16, 16)
    k = randn(3, 3)
    @test_call target_modules = (Contourlets,) Contourlets.conv2d(x, k)
    dst = similar(x)
    @test_call target_modules = (Contourlets,) Contourlets.conv2d!(dst, x, k)
    @test_call target_modules = (Contourlets,) conv2d_sep(x, [1.0, 2.0, 1.0], [1.0, 2.0, 1.0])
    @test_call target_modules = (Contourlets,) conv2d_sep!(dst, x, [1.0, 2.0, 1.0], [1.0, 2.0, 1.0])

    @test_call target_modules = (Contourlets,) rect_downsample(x)
    y = rect_downsample(x)
    @test_call target_modules = (Contourlets,) rect_upsample(y)
    dst2 = zeros(4, 4)
    @test_call target_modules = (Contourlets,) rect_downsample!(dst2, x)
    @test_call target_modules = (Contourlets,) qx_downsample(x)
    qy = qx_downsample(x)
    @test_call target_modules = (Contourlets,) qx_upsample(qy, (16, 16))
    @test_call target_modules = (Contourlets,) shear(x, :h)
    @test_call target_modules = (Contourlets,) shear(x, :v)
    @test_call target_modules = (Contourlets,) inv_shear(x, :h)
    @test_call target_modules = (Contourlets,) inv_shear(x, :v)
    @test_call target_modules = (Contourlets,) shear!(dst, x, :h)
    @test_call target_modules = (Contourlets,) inv_shear!(dst, x, :v)

    # ── Pyramid ──────────────────────────────────────────────────────────────
    @test_call target_modules = (Contourlets,) lp_decompose(x, CDF97)
    c, bp = lp_decompose(x, CDF97)
    @test_call target_modules = (Contourlets,) lp_reconstruct(c, bp, CDF97)
    @test_call target_modules = (Contourlets,) nsp_decompose(x, CDF97, 1)
    nc, nbp = nsp_decompose(x, CDF97, 1)
    @test_call target_modules = (Contourlets,) nsp_reconstruct(nc, nbp, CDF97, 1)

    # ── Directional FB ───────────────────────────────────────────────────────
    @test_call target_modules = (Contourlets,) qfb_decompose(x, Q2345)
    sb0, sb1 = qfb_decompose(x, Q2345)
    @test_call target_modules = (Contourlets,) qfb_reconstruct(sb0, sb1, Q2345)
    @test_call target_modules = (Contourlets,) dfb_decompose(x, 2, Q2345)
    sbs = dfb_decompose(x, 2, Q2345)
    @test_call target_modules = (Contourlets,) dfb_reconstruct(sbs, Q2345)
    @test_call target_modules = (Contourlets,) nsdfb_decompose(x, 2, Q2345, 1)
    nsbs = nsdfb_decompose(x, 2, Q2345, 1)
    @test_call target_modules = (Contourlets,) nsdfb_reconstruct(nsbs, Q2345, 1)

    # ── Transforms ───────────────────────────────────────────────────────────
    p = ContourletParams(J = 2, L_array = [1, 2])
    @test_call target_modules = (Contourlets,) ct_forward(x, p)
    coeffs = ct_forward(x, p)
    @test_call target_modules = (Contourlets,) ct_inverse(coeffs, p)
    @test_call target_modules = (Contourlets,) nsct_forward(x, p)
    nc_coeffs = nsct_forward(x, p)
    @test_call target_modules = (Contourlets,) nsct_inverse(nc_coeffs, p)

    # ── Workspace ────────────────────────────────────────────────────────────
    @test_call target_modules = (Contourlets,) make_workspace(p, (16, 16))
    @test_call target_modules = (Contourlets,) make_nsct_workspace(p, (16, 16))
    ws = make_workspace(p, (16, 16))

    @test_call target_modules = (Contourlets,) similar_coefficients(p, (16, 16))
    @test_call target_modules = (Contourlets,) similar_nsct_coefficients(p, (16, 16))
    sc = similar_coefficients(p, (16, 16))
    @test_call target_modules = (Contourlets,) ct_forward!(sc, x, p; workspace = ws)
    rec = similar(x)
    @test_call target_modules = (Contourlets,) ct_inverse!(rec, sc, p; workspace = ws)
    nsws = make_nsct_workspace(p, (16, 16))
    nsc = similar_nsct_coefficients(p, (16, 16))
    @test_call target_modules = (Contourlets,) nsct_forward!(nsc, x, p; workspace = nsws)
    @test_call target_modules = (Contourlets,) nsct_inverse!(rec, nsc, p; workspace = nsws)

    # ── TICT / STICT ─────────────────────────────────────────────────────────
    @test_call target_modules = (Contourlets,) tict_forward(x, p)
    @test_call target_modules = (Contourlets,) stict_forward(x, p)
    tc, sc2 = tict_forward(x, p), stict_forward(x, p)
    @test_call target_modules = (Contourlets,) tict_inverse(tc, p)
    @test_call target_modules = (Contourlets,) stict_inverse(sc2, p)
    @test_call target_modules = (Contourlets,) make_tict_workspace(p, (16, 16))
    @test_call target_modules = (Contourlets,) make_stict_workspace(p, (16, 16))
    @test_call target_modules = (Contourlets,) similar_tict_coefficients(p, (16, 16))
    @test_call target_modules = (Contourlets,) similar_stict_coefficients(p, (16, 16))
end

@testitem "@test_opt Contourlets core functions" tags = [:jet] begin
    using JET, Random
    Random.seed!(101)
    x = randn(16, 16)
    dst = similar(x)

    # ── Primitives ───────────────────────────────────────────────────────────
    @test_opt target_modules = (Contourlets,) rect_downsample(x)
    @test_opt target_modules = (Contourlets,) shear(x, :h)
    @test_opt target_modules = (Contourlets,) shear!(dst, x, :v)
    @test_opt target_modules = (Contourlets,) inv_shear(x, :h)
    @test_opt target_modules = (Contourlets,) conv2d_sep(x, [1.0, 2.0, 1.0], [1.0, 2.0, 1.0])

    # ── Pyramid ──────────────────────────────────────────────────────────────
    @test_opt target_modules = (Contourlets,) lp_decompose(x, CDF97)
    c, bp = lp_decompose(x, CDF97)
    @test_opt target_modules = (Contourlets,) lp_reconstruct(c, bp, CDF97)
    @test_opt target_modules = (Contourlets,) nsp_decompose(x, CDF97, 1)

    # ── Directional FB ───────────────────────────────────────────────────────
    @test_opt target_modules = (Contourlets,) qfb_decompose(x, Q2345)
    sb0, sb1 = qfb_decompose(x, Q2345)
    @test_opt target_modules = (Contourlets,) qfb_reconstruct(sb0, sb1, Q2345)
    @test_opt target_modules = (Contourlets,) dfb_decompose(x, 2, Q2345)
    sbs = dfb_decompose(x, 2, Q2345)
    @test_opt target_modules = (Contourlets,) dfb_reconstruct(sbs, Q2345)

    # ── Transforms ───────────────────────────────────────────────────────────
    p = ContourletParams(J = 2, L_array = [1, 2])
    @test_opt target_modules = (Contourlets,) ct_forward(x, p)
    coeffs = ct_forward(x, p)
    @test_opt target_modules = (Contourlets,) ct_inverse(coeffs, p)
    @test_opt target_modules = (Contourlets,) nsct_forward(x, p)
    nc = nsct_forward(x, p)
    @test_opt target_modules = (Contourlets,) nsct_inverse(nc, p)

    # ── Workspace API ────────────────────────────────────────────────────────
    ws = make_workspace(p, (16, 16))
    sc = similar_coefficients(p, (16, 16))
    @test_opt target_modules = (Contourlets,) ct_forward!(sc, x, p; workspace = ws)
    rec = similar(x)
    @test_opt target_modules = (Contourlets,) ct_inverse!(rec, sc, p; workspace = ws)

    # The TICT/STICT workspace paths are the hot loops: no dynamic dispatch.
    tws, sws = make_tict_workspace(p, (16, 16)), make_stict_workspace(p, (16, 16))
    tc, stc = similar_tict_coefficients(p, (16, 16)), similar_stict_coefficients(p, (16, 16))
    @test_opt target_modules = (Contourlets,) tict_forward!(tc, x, p; workspace = tws)
    @test_opt target_modules = (Contourlets,) tict_inverse!(rec, tc, p; workspace = tws)
    @test_opt target_modules = (Contourlets,) stict_forward!(stc, x, p; workspace = sws)
    @test_opt target_modules = (Contourlets,) stict_inverse!(rec, stc, p; workspace = sws)
end

@testitem "@test_call Contourlets in-place and threading API" tags = [:jet] begin
    using JET, Random
    Random.seed!(102)
    M = (Contourlets,)
    x = randn(16, 16)
    p = ContourletParams(J = 2, L_array = [1, 2])

    # ── Threading policies ───────────────────────────────────────────────────
    @test_call target_modules = M Auto()
    @test_call target_modules = M Enabled()
    @test_call target_modules = M Disabled()
    @test_call target_modules = M FilterPair(CDF97.h, CDF97.g)
    @test_call target_modules = M ct_forward(x, p; threading = Disabled())
    @test_call target_modules = M nsct_forward(x, p; threading = Enabled())

    # ── Primitives (in place) ────────────────────────────────────────────────
    y = rect_downsample(x)
    @test_call target_modules = M rect_upsample!(similar(x), y)
    qy = qx_downsample(x)
    @test_call target_modules = M qx_downsample!(similar(qy), x)
    @test_call target_modules = M qx_upsample!(similar(x), qy)

    # ── Pyramids (in place) ──────────────────────────────────────────────────
    c, bp = lp_decompose(x, CDF97)
    @test_call target_modules = M lp_decompose!(similar(c), similar(bp), x, CDF97)
    @test_call target_modules = M lp_reconstruct!(similar(x), c, bp, CDF97)
    nc, nbp = nsp_decompose(x, CDF97, 1)
    @test_call target_modules = M nsp_decompose!(similar(nc), similar(nbp), x, CDF97, 1)
    @test_call target_modules = M nsp_reconstruct!(similar(x), nc, nbp, CDF97, 1)

    # ── Directional filter banks (in place) ──────────────────────────────────
    sb0, sb1 = qfb_decompose(x, Q2345)
    @test_call target_modules = M qfb_decompose!(similar(sb0), similar(sb1), x, Q2345)
    @test_call target_modules = M qfb_reconstruct!(similar(x), sb0, sb1, Q2345)
    @test_call target_modules = M dfb_subband_sizes(16, 16, 2)
    sbs = dfb_decompose(x, 2, Q2345)
    @test_call target_modules = M dfb_decompose!([similar(s) for s in sbs], x, 2, Q2345)
    @test_call target_modules = M dfb_reconstruct!(similar(x), sbs, Q2345)

    # ── Coefficient / workspace types (through their constructors) ───────────
    ws = make_workspace(p, (16, 16))
    @test_call target_modules = M similar_coefficients(p, (16, 16))
    @test_call target_modules = M make_nsct_workspace(p, (16, 16))
    @test_call target_modules = M similar_nsct_coefficients(p, (16, 16))
    @test ws isa CTWorkspace
    @test similar_coefficients(p, (16, 16)) isa ContourletCoefficients
    @test similar_nsct_coefficients(p, (16, 16)) isa NSCTCoefficients
    @test make_nsct_workspace(p, (16, 16)) isa NSCTWorkspace
    @test similar_tict_coefficients(p, (16, 16)) isa TICTCoefficients
    @test similar_stict_coefficients(p, (16, 16)) isa TICTCoefficients
    @test make_tict_workspace(p, (16, 16)) isa TICTWorkspace
    @test make_stict_workspace(p, (16, 16)) isa STICTWorkspace

    # ── TICT / STICT (in place) ──────────────────────────────────────────────
    tws, sws = make_tict_workspace(p, (16, 16)), make_stict_workspace(p, (16, 16))
    tc, stc = similar_tict_coefficients(p, (16, 16)), similar_stict_coefficients(p, (16, 16))
    rec = similar(x)
    @test_call target_modules = M tict_forward!(tc, x, p; workspace = tws)
    @test_call target_modules = M tict_forward!(tc, x, p; workspace = tws, threading = Disabled())
    @test_call target_modules = M tict_inverse!(rec, tc, p; workspace = tws)
    @test_call target_modules = M stict_forward!(stc, x, p; workspace = sws)
    @test_call target_modules = M stict_inverse!(rec, stc, p; workspace = sws)
    @test_call target_modules = M nsct_forward!(
        similar_nsct_coefficients(p, (16, 16)), x, p; workspace = make_nsct_workspace(p, (16, 16)), threading = Disabled()
    )
end

@testitem "@test_opt Contourlets remaining API" tags = [:jet] begin
    using JET, Random
    Random.seed!(103)
    M = (Contourlets,)
    x = randn(16, 16)
    k = randn(3, 3)
    h = [1.0, 2.0, 1.0]
    dst = similar(x)
    p = ContourletParams(J = 2, L_array = [1, 2])

    # ── Types / constructors / filters ───────────────────────────────────────
    @test_opt target_modules = M FilterPair([1.0, 2.0], [1.0, 2.0])
    @test_opt target_modules = M QuincunxFilterPair([0.25, 0.25])
    @test_opt target_modules = M ContourletParams(J = 2, L_array = [1, 2])
    @test_opt target_modules = M parabolic_levels(4)
    @test_opt target_modules = M upsample_filter([1.0, 2.0, 3.0], 2)
    @test_opt target_modules = M upsample_kernel([1.0 2.0; 3.0 4.0], 2)
    @test_opt target_modules = M check_pr_condition(Q2345)
    @test_opt target_modules = M Auto()
    @test_opt target_modules = M Enabled()
    @test_opt target_modules = M Disabled()

    # ── Primitives ───────────────────────────────────────────────────────────
    @test_opt target_modules = M Contourlets.conv2d(x, k)
    @test_opt target_modules = M Contourlets.conv2d!(dst, x, k)
    @test_opt target_modules = M conv2d_sep!(dst, x, h, h)
    y = rect_downsample(x)
    qy = qx_downsample(x)
    @test_opt target_modules = M rect_downsample!(similar(y), x)
    @test_opt target_modules = M rect_upsample(y)
    @test_opt target_modules = M rect_upsample!(similar(x), y)
    @test_opt target_modules = M qx_downsample(x)
    @test_opt target_modules = M qx_downsample!(similar(qy), x)
    @test_opt target_modules = M qx_upsample(qy, (16, 16))
    @test_opt target_modules = M qx_upsample!(similar(x), qy)
    @test_opt target_modules = M inv_shear!(dst, x, :h)

    # ── Pyramids ─────────────────────────────────────────────────────────────
    c, bp = lp_decompose(x, CDF97)
    @test_opt target_modules = M lp_decompose!(similar(c), similar(bp), x, CDF97)
    @test_opt target_modules = M lp_reconstruct!(similar(x), c, bp, CDF97)
    nc, nbp = nsp_decompose(x, CDF97, 1)
    @test_opt target_modules = M nsp_decompose!(similar(nc), similar(nbp), x, CDF97, 1)
    @test_opt target_modules = M nsp_reconstruct(nc, nbp, CDF97, 1)
    @test_opt target_modules = M nsp_reconstruct!(similar(x), nc, nbp, CDF97, 1)

    # ── Directional filter banks ─────────────────────────────────────────────
    sb0, sb1 = qfb_decompose(x, Q2345)
    @test_opt target_modules = M qfb_decompose!(similar(sb0), similar(sb1), x, Q2345)
    @test_opt target_modules = M qfb_reconstruct!(similar(x), sb0, sb1, Q2345)
    @test_opt target_modules = M dfb_subband_sizes(16, 16, 2)
    sbs = dfb_decompose(x, 2, Q2345)
    @test_opt target_modules = M dfb_decompose!([similar(s) for s in sbs], x, 2, Q2345)
    @test_opt target_modules = M dfb_reconstruct!(similar(x), sbs, Q2345)
    @test_opt target_modules = M nsdfb_decompose(x, 2, Q2345, 1)
    nsbs = nsdfb_decompose(x, 2, Q2345, 1)
    @test_opt target_modules = M nsdfb_reconstruct(nsbs, Q2345, 1)

    # ── Workspaces, coefficient containers and the remaining in-place paths ──
    # Known runtime dispatch in the allocating builders: they pass type-valued keywords
    # (`T`, `Td`, `M`: `Type{<:Number}` / `DataType`) through `similar_*coefficients`, which cannot
    # specialise on them, and `make_*workspace` runs a warm-up through that path.  The in-place
    # `!` paths that consume the workspaces are checked above and below without `broken`.
    @test_opt broken = true target_modules = M make_workspace(p, (16, 16))
    @test_opt broken = true target_modules = M make_nsct_workspace(p, (16, 16))
    @test_opt target_modules = M make_tict_workspace(p, (16, 16))
    @test_opt target_modules = M make_stict_workspace(p, (16, 16))
    @test_opt target_modules = M similar_coefficients(p, (16, 16))
    @test_opt target_modules = M similar_nsct_coefficients(p, (16, 16))
    @test_opt target_modules = M similar_tict_coefficients(p, (16, 16))
    @test_opt target_modules = M similar_stict_coefficients(p, (16, 16))
    nsws = make_nsct_workspace(p, (16, 16))
    nsc = similar_nsct_coefficients(p, (16, 16))
    @test_opt target_modules = M nsct_forward!(nsc, x, p; workspace = nsws)
    @test_opt target_modules = M nsct_inverse!(dst, nsc, p; workspace = nsws)
    @test_opt broken = true target_modules = M tict_forward(x, p)    # builds its own workspace (same cause)
    @test_opt broken = true target_modules = M stict_forward(x, p)
    @test_opt target_modules = M tict_inverse!(dst, similar_tict_coefficients(p, (16, 16)), p; workspace = make_tict_workspace(p, (16, 16)))
    tcs = tict_forward(x, p)
    scs = stict_forward(x, p)
    @test_opt broken = true target_modules = M tict_inverse(tcs, p)
    @test_opt broken = true target_modules = M stict_inverse(scs, p)
end
