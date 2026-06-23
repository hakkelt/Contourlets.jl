# Universal (backend-agnostic) GPU tests for the ContourletsGPUExt extension.
#
# GPUEnv enumerates the GPU backends worth trying on the current host: JLArrays
# (a CPU-side mock, always available, so these tests run in ordinary CI) plus any
# real backend (CUDA, AMDGPU, Metal, oneAPI, OpenCL) that is installed and
# functional.  The same test body therefore validates every available backend.
#
# Loading a backend (JLArrays / CUDA) pulls in GPUArrays + KernelAbstractions,
# which triggers the package extension.

@testmodule GPUBackends begin
    using GPUEnv
    export backends

    # All backends to exercise (JLArrays always present).  GPUEnv supplies the
    # backend packages from its overlay environment, so the tests need no direct
    # GPUArrays / KernelAbstractions / JLArrays dependency.
    const backends = gpu_backends(; include_jlarrays = true)
end

@testitem "GPU at least one backend (JLArrays)" tags = [:gpu] setup = [GPUBackends] begin
    @test !isempty(GPUBackends.backends)
    @test any(b -> b.name === :JLArray, GPUBackends.backends)
end

@testitem "GPU primitives match CPU" tags = [:gpu] setup = [GPUBackends] begin
    using GPUEnv, Random
    Random.seed!(50)
    x = randn(32, 32)
    h = [0.25, 0.5, 0.25]
    for backend in GPUBackends.backends
        xg = to_gpu(backend, x)
        # separable convolution, both boundaries
        for bnd in (:symmetric, :periodic)
            dg = similar(xg)
            Contourlets.conv2d_sep!(dg, xg, h, h; boundary = bnd)
            @test maximum(abs, Array(dg) .- Contourlets.conv2d_sep(x, h, h; boundary = bnd)) < 1.0e-12
        end
        # shear / inv_shear round-trip and CPU match
        for dir in (:h, :v)
            sg = shear(xg, dir)
            @test maximum(abs, Array(sg) .- shear(x, dir)) < 1.0e-12
            @test maximum(abs, Array(inv_shear(sg, dir)) .- x) < 1.0e-12
        end
        # rectangular down/up-sampling
        dn = rect_downsample(xg)
        @test maximum(abs, Array(dn) .- rect_downsample(x)) < 1.0e-12
        up = rect_upsample(dn)
        @test maximum(abs, Array(up) .- rect_upsample(Array(dn))) < 1.0e-12
        # quincunx down/up-sampling
        qd = qx_downsample(xg)
        @test maximum(abs, Array(qd) .- qx_downsample(x)) < 1.0e-12
    end
end

@testitem "GPU Laplacian pyramid matches CPU + PR" tags = [:gpu] setup = [GPUBackends] begin
    using GPUEnv, Random
    Random.seed!(51)
    x = randn(32, 32)
    cc, bpc = lp_decompose(x, CDF97)
    for backend in GPUBackends.backends
        xg = to_gpu(backend, x)
        cg, bpg = lp_decompose(xg, CDF97)
        @test maximum(abs, Array(cg) .- cc) < 1.0e-12
        @test maximum(abs, Array(bpg) .- bpc) < 1.0e-12
        rec = lp_reconstruct(cg, bpg, CDF97)
        @test maximum(abs, Array(rec) .- x) < 1.0e-12
    end
end

@testitem "GPU nonsubsampled pyramid matches CPU + PR" tags = [:gpu] setup = [GPUBackends] begin
    using GPUEnv, Random
    Random.seed!(52)
    x = randn(32, 32)
    for backend in GPUBackends.backends
        xg = to_gpu(backend, x)
        for lvl in 1:2
            cc, bpc = nsp_decompose(x, CDF97, lvl)
            cg, bpg = nsp_decompose(xg, CDF97, lvl)
            @test maximum(abs, Array(cg) .- cc) < 1.0e-12
            @test maximum(abs, Array(bpg) .- bpc) < 1.0e-12
            rec = nsp_reconstruct(cg, bpg, CDF97, lvl)
            @test maximum(abs, Array(rec) .- x) < 1.0e-12
        end
    end
end

@testitem "GPU CT matches CPU and reconstructs" tags = [:gpu] setup = [GPUBackends] begin
    using GPUEnv, Random
    Random.seed!(53)
    x = randn(64, 64)
    p = ContourletParams(J = 2, L_array = [2, 3])
    ccpu = ct_forward(x, p)
    for backend in GPUBackends.backends
        xg = to_gpu(backend, x)
        cgpu = ct_forward(xg, p)
        # Coefficients stay on the device; bring them back to compare to the CPU path.
        @test maximum(abs, Array(cgpu.coarse) .- ccpu.coarse) < 1.0e-12
        for j in 1:2, k in eachindex(ccpu.subbands[j])
            @test maximum(abs, Array(cgpu.subbands[j][k]) .- ccpu.subbands[j][k]) < 1.0e-12
        end
        rec = ct_inverse(cgpu, p)
        @test maximum(abs, Array(rec) .- x) < 1.0e-11
    end
end

@testitem "GPU NSCT matches CPU, reconstructs, shift-invariant" tags = [:gpu] setup = [GPUBackends] begin
    using GPUEnv, Random
    Random.seed!(54)
    x = randn(32, 32)
    p = ContourletParams(J = 2, L_array = [2, 3])
    ncpu = nsct_forward(x, p)
    for backend in GPUBackends.backends
        xg = to_gpu(backend, x)
        ngpu = nsct_forward(xg, p)
        @test maximum(abs, Array(ngpu.coarse) .- ncpu.coarse) < 1.0e-12
        for j in 1:2, k in eachindex(ncpu.subbands[j])
            @test maximum(abs, Array(ngpu.subbands[j][k]) .- ncpu.subbands[j][k]) < 1.0e-12
        end
        rec = nsct_inverse(ngpu, p)
        @test maximum(abs, Array(rec) .- x) < 1.0e-11
        # Shift invariance on device.
        s = (3, 5)
        nshift = nsct_forward(to_gpu(backend, circshift(x, s)), p)
        @test maximum(abs, nshift.subbands[2][1] .- circshift(ngpu.subbands[2][1], s)) < 1.0e-11
    end
end

@testitem "GPU complex CT/NSCT match CPU (real filters on device)" tags = [:gpu] setup = [GPUBackends] begin
    using GPUEnv, Random
    Random.seed!(55)
    z = complex.(randn(32, 32), randn(32, 32))
    p = ContourletParams(J = 2, L_array = [2, 3])
    ccpu = ct_forward(z, p)
    ncpu = nsct_forward(z, p)
    for backend in GPUBackends.backends
        zg = to_gpu(backend, z)

        cgpu = ct_forward(zg, p)
        @test eltype(cgpu.coarse) <: Complex
        @test maximum(abs, Array(cgpu.coarse) .- ccpu.coarse) < 1.0e-12
        @test maximum(abs, Array(ct_inverse(cgpu, p)) .- z) < 1.0e-11

        ngpu = nsct_forward(zg, p)
        @test eltype(ngpu.coarse) <: Complex
        @test maximum(abs, Array(ngpu.coarse) .- ncpu.coarse) < 1.0e-12
        @test maximum(abs, Array(nsct_inverse(ngpu, p)) .- z) < 1.0e-11
    end
end

@testitem "GPU coefficients stay on the device" tags = [:gpu] setup = [GPUBackends] begin
    using GPUEnv, Random
    Random.seed!(56)
    x = randn(32, 32)
    p = ContourletParams(J = 2, L_array = [2, 3])
    for backend in GPUBackends.backends
        xg = to_gpu(backend, x)
        DT = typeof(xg)

        c = ct_forward(xg, p)
        @test !(c.coarse isa Array)            # not transferred to host
        @test c.coarse isa DT
        @test all(s isa DT for lvl in c.subbands for s in lvl)
        r = ct_inverse(c, p)
        @test !(r isa Array)
        @test maximum(abs, Array(r) .- x) < 1.0e-11

        nc = nsct_forward(xg, p)
        @test !(nc.coarse isa Array)
        @test all(s isa DT for lvl in nc.subbands for s in lvl)
        nr = nsct_inverse(nc, p)
        @test !(nr isa Array)
        @test maximum(abs, Array(nr) .- x) < 1.0e-11
    end
end

@testitem "GPU primitives error paths" tags = [:gpu] setup = [GPUBackends] begin
    using GPUEnv
    x = randn(16, 16)
    for backend in GPUBackends.backends
        xg = to_gpu(backend, x)
        dg = similar(xg)
        h = [0.25, 0.5, 0.25]
        @test_throws ArgumentError Contourlets.conv2d_sep!(dg, xg, h, h; boundary = :bad)
        @test_throws ArgumentError Contourlets.shear!(dg, xg, :bad)
        @test_throws ArgumentError Contourlets.inv_shear!(dg, xg, :bad)
        @test_throws ArgumentError Contourlets._resamp(xg, 5)
        @test_throws ArgumentError Contourlets._sefilter2(xg, Float64.(h), 0, 0, :bad)
    end
end

@testitem "GPU qx_upsample" tags = [:gpu] setup = [GPUBackends] begin
    using GPUEnv, Random
    Random.seed!(57)
    x = randn(16, 16)
    for backend in GPUBackends.backends
        xg = to_gpu(backend, x)
        qd = qx_downsample(xg)
        qu = qx_upsample(qd, size(xg))
        @test size(qu) == size(xg)
        # upsample should reverse the downsample on even+even sites
        qu_cpu = qx_upsample(qx_downsample(x), size(x))
        @test maximum(abs, Array(qu) .- qu_cpu) < 1.0e-12
    end
end

@testitem "GPU nsdfb edge cases and error paths" tags = [:gpu] setup = [GPUBackends] begin
    using GPUEnv, Random
    Random.seed!(58)
    x = randn(16, 16)
    p = ContourletParams(J = 1, L_array = [2])
    for backend in GPUBackends.backends
        xg = to_gpu(backend, x)
        # l_levels == 0 → [copy(bandpass)]
        sbs0 = nsdfb_decompose(xg, 0, Q2345)
        @test length(sbs0) == 1
        @test maximum(abs, Array(sbs0[1]) .- x) < 1.0e-12
        # nsdfb_reconstruct n == 1 → copy
        rec1 = nsdfb_reconstruct([xg], Q2345)
        @test maximum(abs, Array(rec1) .- x) < 1.0e-12
        # error: l_levels < 0
        @test_throws ArgumentError nsdfb_decompose(xg, -1, Q2345)
        # error: non-power-of-2 subbands
        sbs2 = nsdfb_decompose(xg, 2, Q2345)
        @test_throws ArgumentError nsdfb_reconstruct(sbs2[1:3], Q2345)
    end
end

@testitem "GPU qfb_decompose/reconstruct non-ladder filter" tags = [:gpu] setup = [GPUBackends] begin
    using GPUEnv, Random
    Random.seed!(59)
    x = randn(16, 16)
    # Haar-like non-ladder QuincunxFilterPair
    haar = QuincunxFilterPair([0.5 0.5], [1.0 1.0], (1, 1), (1, 2))
    for backend in GPUBackends.backends
        xg = to_gpu(backend, x)
        # :col direction — GPU kernel path
        sb0g, sb1g = qfb_decompose(xg, haar; dir = :col)
        sb0c, sb1c = qfb_decompose(x, haar; dir = :col)
        @test maximum(abs, Array(sb0g) .- sb0c) < 1.0e-12
        @test maximum(abs, Array(sb1g) .- sb1c) < 1.0e-12
        # GPU results must stay on device
        @test !(sb0g isa Array)
        # synthesis round-trip (col)
        rec_col = qfb_reconstruct(sb0g, sb1g, haar; dir = :col)
        @test !(rec_col isa Array)
        # :row direction — transpose-to-CPU-and-back path
        sb0r, sb1r = qfb_decompose(xg, haar; dir = :row)
        @test !(sb0r isa Array)
        rec_row = qfb_reconstruct(sb0r, sb1r, haar; dir = :row)
        @test !(rec_row isa Array)
        @test maximum(abs, Array(rec_row) .- x) < 1.0e-12
        # ladder filter throws
        @test_throws ArgumentError qfb_decompose(xg, Q2345)
        @test_throws ArgumentError qfb_reconstruct(sb0g, sb1g, Q2345)
    end
end

@testitem "GPU DFB with non-ladder filter (exercises shear kernels)" tags = [:gpu] setup = [GPUBackends] begin
    using GPUEnv, Random
    Random.seed!(73)
    x = randn(16, 16)
    # Haar-like non-ladder pair: is_ladder == false → _dfb_split path → shear!/inv_shear!
    haar = QuincunxFilterPair([0.5 0.5], [1.0 1.0], (1, 1), (1, 2))
    for backend in GPUBackends.backends
        xg = to_gpu(backend, x)
        # l=2 forces depth 1 (:h shear) and depth 2 (:v shear), hitting all 4 kernels
        sbs = dfb_decompose(xg, 2, haar)
        @test length(sbs) == 4
        @test !(sbs[1] isa Array)
        # CPU reference
        sbs_cpu = dfb_decompose(x, 2, haar)
        for k in 1:4
            @test maximum(abs, Array(sbs[k]) .- sbs_cpu[k]) < 1.0e-12
        end
        rec = dfb_reconstruct(sbs, haar)
        @test !(rec isa Array)
        @test maximum(abs, Array(rec) .- x) < 1.0e-12
    end
end

@testitem "GPU qfb_decompose boundary clamping" tags = [:gpu] setup = [GPUBackends] begin
    using GPUEnv, Random
    Random.seed!(74)
    x = randn(8, 8)
    # 5-tap filter centred at position 3: for the first output column (j=2),
    # tap l=5 gives jj = 2-2 = 0 < 1; for the last column (j=8), tap l=1
    # gives jj = 8+2 = 10 > 8.  Both clamping branches are reached.
    h5 = Float64[1 / 16 1 / 4 3 / 8 1 / 4 1 / 16]   # 1×5
    wide_qfp = QuincunxFilterPair(h5, h5, (1, 3), (1, 3))
    for backend in GPUBackends.backends
        xg = to_gpu(backend, x)
        sb0, sb1 = qfb_decompose(xg, wide_qfp; dir = :col)
        @test !(sb0 isa Array)
        @test size(sb0) == (8, 4)
        @test all(isfinite, Array(sb0))
    end
end

@testitem "GPU make_nsct_workspace migrates filter caches to device" tags = [:gpu] setup = [GPUBackends] begin
    using GPUEnv, Random
    Random.seed!(75)
    p = ContourletParams(J = 1, L_array = [2])
    x = randn(32, 32)
    for backend in GPUBackends.backends
        xg = to_gpu(backend, x)
        # make_nsct_workspace(image, params) infers M from image → dispatches
        # _device_qup_cache and _device_lp_cache GPU overloads
        ws = make_nsct_workspace(xg, p; prewarm = false)
        @test ws isa ContourletWorkspace
        # NSDFB filter bundles should now be on the device
        @test !(ws.qup_cache[1].k1 isa Array)
        # à-trous LP filter vectors should also be on the device
        @test !(ws.lp_h_cache[1] isa Array)
        @test !(ws.lp_g_cache[1] isa Array)
    end
end

@testitem "GPU Adapt.adapt_structure for CT/NSCT coefficients" tags = [:gpu] setup = [GPUBackends] begin
    using GPUEnv, Random
    # Adapt is loaded transitively by JLArrays / KernelAbstractions (GPUEnv dep).
    const Adapt = Base.require(
        Base.PkgId(Base.UUID("79e6a3ab-5dfb-504d-930d-738a2a938a0e"), "Adapt")
    )
    Random.seed!(60)
    x = randn(32, 32)
    p = ContourletParams(J = 2, L_array = [2, 3])
    for backend in GPUBackends.backends
        xg = to_gpu(backend, x)
        # CT: move CPU coeffs to device and back via adapt_structure
        cc = ct_forward(x, p)
        cc_dev = Adapt.adapt(typeof(xg), cc)
        @test !(cc_dev.coarse isa Array)
        cc_host = Adapt.adapt(Array, cc_dev)
        @test cc_host.coarse isa Array
        @test maximum(abs, cc_host.coarse .- cc.coarse) < 1.0e-12
        # NSCT: same round-trip
        nc = nsct_forward(x, p)
        nc_dev = Adapt.adapt(typeof(xg), nc)
        @test !(nc_dev.coarse isa Array)
        nc_host = Adapt.adapt(Array, nc_dev)
        @test nc_host.coarse isa Array
        @test maximum(abs, nc_host.coarse .- nc.coarse) < 1.0e-12
    end
end
