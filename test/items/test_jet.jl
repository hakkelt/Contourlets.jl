@testitem "@test_call Contourlets public API" tags = [:jet] begin
    using JET, Random
    Random.seed!(100)

    # ── Types / constructors ─────────────────────────────────────────────────
    @test_call target_modules = (Contourlets,) FilterPair([1.0, 2.0], [1.0, 2.0])
    @test_call target_modules = (Contourlets,) QuincunxFilterPair([0.5 0.5], [1.0 1.0], (1, 1), (1, 2))
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

    @test_call target_modules = (Contourlets,) estimate_workspace_size(p, (16, 16))
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
end
