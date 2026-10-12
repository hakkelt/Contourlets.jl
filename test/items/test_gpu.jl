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
    using Adapt
    export backends, fft_backends, Adapt

    # `test/runtests.jl` activates the GPU overlay before the runner starts, but a
    # test item can also be run on its own — an IDE test runner, or a language
    # server's test tooling — and that starts in the plain test environment where
    # no backend package is loadable and `gpu_backends()` comes back empty.
    # Activating here keeps the items self-contained either way; when the overlay
    # already exists (`persist = true` writes it to `test/gpu_env`) this only
    # loads it.
    isempty(gpu_backends(; include_jlarrays = true)) &&
        GPUEnv.activate(persist = true, include_jlarrays = true)

    # All backends to exercise (JLArrays always present).  GPUEnv supplies the
    # backend packages from its overlay environment, so the tests need no direct
    # GPUArrays / KernelAbstractions / JLArrays dependency.
    # All test data below is Float32 (Metal does not support Float64), so every
    # backend — including Metal — can run the same test bodies.
    const backends = gpu_backends(; include_jlarrays = true)

    # The directional filter banks filter on the DFT grid, so the transforms need a
    # backend with an FFT (cuFFT, rocFFT, …, or FFTW on JLArrays' host memory);
    # OpenCL has none.
    const fft_backends = gpu_backends(; include_jlarrays = true, supports_fftw = true)
end

@testitem "GPU at least one backend (JLArrays)" tags = [:gpu] setup = [GPUBackends] begin
    @test !isempty(GPUBackends.backends)
    @test any(b -> b.name === :JLArray, GPUBackends.backends)
end

@testitem "GPU primitives match CPU" tags = [:gpu] setup = [GPUBackends] begin
    using GPUEnv, Random
    Random.seed!(50)
    x = randn(Float32, 32, 32)
    h = Float32[0.25, 0.5, 0.25]
    for backend in GPUBackends.backends
        xg = to_gpu(backend, x)
        # separable convolution, both boundaries
        for bnd in (:symmetric, :periodic)
            dg = similar(xg)
            Contourlets.conv2d_sep!(dg, xg, h, h; boundary = bnd)
            @test maximum(abs, Array(dg) .- Contourlets.conv2d_sep(x, h, h; boundary = bnd)) < 1.0f-5
        end
        # shear / inv_shear round-trip and CPU match
        for dir in (:h, :v)
            sg = shear(xg, dir)
            @test maximum(abs, Array(sg) .- shear(x, dir)) < 1.0f-5
            @test maximum(abs, Array(inv_shear(sg, dir)) .- x) < 1.0f-5
        end
        # rectangular down/up-sampling
        dn = rect_downsample(xg)
        @test maximum(abs, Array(dn) .- rect_downsample(x)) < 1.0f-5
        up = rect_upsample(dn)
        @test maximum(abs, Array(up) .- rect_upsample(Array(dn))) < 1.0f-5
        # quincunx down/up-sampling
        qd = qx_downsample(xg)
        @test maximum(abs, Array(qd) .- qx_downsample(x)) < 1.0f-5
    end
end

@testitem "GPU Laplacian pyramid matches CPU + PR" tags = [:gpu] setup = [GPUBackends] begin
    using GPUEnv, Random
    Random.seed!(51)
    x = randn(Float32, 32, 32)
    cc, bpc = lp_decompose(x, CDF97)
    for backend in GPUBackends.backends
        xg = to_gpu(backend, x)
        cg, bpg = lp_decompose(xg, CDF97)
        @test maximum(abs, Array(cg) .- cc) < 1.0f-5
        @test maximum(abs, Array(bpg) .- bpc) < 1.0f-5
        rec = lp_reconstruct(cg, bpg, CDF97)
        @test maximum(abs, Array(rec) .- x) < 1.0f-4
    end
end

@testitem "GPU nonsubsampled pyramid matches CPU + PR" tags = [:gpu] setup = [GPUBackends] begin
    using GPUEnv, Random
    Random.seed!(52)
    x = randn(Float32, 32, 32)
    for backend in GPUBackends.backends
        xg = to_gpu(backend, x)
        for lvl in 1:2
            cc, bpc = nsp_decompose(x, CDF97, lvl)
            cg, bpg = nsp_decompose(xg, CDF97, lvl)
            @test maximum(abs, Array(cg) .- cc) < 1.0f-5
            @test maximum(abs, Array(bpg) .- bpc) < 1.0f-5
            rec = nsp_reconstruct(cg, bpg, CDF97, lvl)
            @test maximum(abs, Array(rec) .- x) < 1.0f-4
        end
    end
end

@testitem "GPU CT matches CPU and reconstructs" tags = [:gpu] setup = [GPUBackends] begin
    using GPUEnv, Random
    Random.seed!(53)
    x = randn(Float32, 64, 64)
    p = ContourletParams(J = 2, L_array = [2, 3])
    ccpu = ct_forward(x, p)
    for backend in GPUBackends.fft_backends
        xg = to_gpu(backend, x)
        cgpu = ct_forward(xg, p)
        # Coefficients stay on the device; bring them back to compare to the CPU path.
        @test maximum(abs, Array(cgpu.coarse) .- ccpu.coarse) < 1.0f-5
        for j in 1:2, k in eachindex(ccpu.subbands[j])
            @test maximum(abs, Array(cgpu.subbands[j][k]) .- ccpu.subbands[j][k]) < 1.0f-5
        end
        rec = ct_inverse(cgpu, p)
        @test maximum(abs, Array(rec) .- x) < 1.0f-4
    end
end

@testitem "GPU NSCT matches CPU, reconstructs, shift-invariant" tags = [:gpu] setup = [GPUBackends] begin
    using GPUEnv, Random
    Random.seed!(54)
    x = randn(Float32, 32, 32)
    p = ContourletParams(J = 2, L_array = [2, 3])
    ncpu = nsct_forward(x, p)
    for backend in GPUBackends.fft_backends
        xg = to_gpu(backend, x)
        ngpu = nsct_forward(xg, p)
        @test maximum(abs, Array(ngpu.coarse) .- ncpu.coarse) < 1.0f-5
        for j in 1:2, k in eachindex(ncpu.subbands[j])
            @test maximum(abs, Array(ngpu.subbands[j][k]) .- ncpu.subbands[j][k]) < 1.0f-5
        end
        rec = nsct_inverse(ngpu, p)
        @test maximum(abs, Array(rec) .- x) < 1.0f-4
        # Shift invariance on device.
        s = (3, 5)
        nshift = nsct_forward(to_gpu(backend, circshift(x, s)), p)
        @test maximum(abs, nshift.subbands[2][1] .- circshift(ngpu.subbands[2][1], s)) < 1.0f-4
    end
end

@testitem "GPU complex CT/NSCT match CPU (real filters on device)" tags = [:gpu] setup = [GPUBackends] begin
    using GPUEnv, Random
    Random.seed!(55)
    z = complex.(randn(Float32, 32, 32), randn(Float32, 32, 32))
    p = ContourletParams(J = 2, L_array = [2, 3])
    ccpu = ct_forward(z, p)
    ncpu = nsct_forward(z, p)
    for backend in GPUBackends.fft_backends
        zg = to_gpu(backend, z)

        cgpu = ct_forward(zg, p)
        @test eltype(cgpu.coarse) <: Complex
        @test maximum(abs, Array(cgpu.coarse) .- ccpu.coarse) < 1.0f-5
        @test maximum(abs, Array(ct_inverse(cgpu, p)) .- z) < 1.0f-4

        ngpu = nsct_forward(zg, p)
        @test eltype(ngpu.coarse) <: Complex
        @test maximum(abs, Array(ngpu.coarse) .- ncpu.coarse) < 1.0f-5
        @test maximum(abs, Array(nsct_inverse(ngpu, p)) .- z) < 1.0f-4
    end
end

@testitem "GPU coefficients stay on the device" tags = [:gpu] setup = [GPUBackends] begin
    using GPUEnv, Random
    Random.seed!(56)
    x = randn(Float32, 32, 32)
    p = ContourletParams(J = 2, L_array = [2, 3])
    for backend in GPUBackends.fft_backends
        xg = to_gpu(backend, x)
        DT = typeof(xg)

        c = ct_forward(xg, p)
        @test !(c.coarse isa Array)            # not transferred to host
        @test c.coarse isa DT
        @test all(s isa DT for lvl in c.subbands for s in lvl)
        r = ct_inverse(c, p)
        @test !(r isa Array)
        @test maximum(abs, Array(r) .- x) < 1.0f-4

        nc = nsct_forward(xg, p)
        @test !(nc.coarse isa Array)
        @test all(s isa DT for lvl in nc.subbands for s in lvl)
        nr = nsct_inverse(nc, p)
        @test !(nr isa Array)
        @test maximum(abs, Array(nr) .- x) < 1.0f-4
    end
end

@testitem "GPU primitives error paths" tags = [:gpu] setup = [GPUBackends] begin
    using GPUEnv
    x = randn(Float32, 16, 16)
    for backend in GPUBackends.backends
        xg = to_gpu(backend, x)
        dg = similar(xg)
        h = Float32[0.25, 0.5, 0.25]
        @test_throws ArgumentError Contourlets.conv2d_sep!(dg, xg, h, h; boundary = :bad)
        @test_throws ArgumentError Contourlets.shear!(dg, xg, :bad)
        @test_throws ArgumentError Contourlets.inv_shear!(dg, xg, :bad)
    end
end

@testitem "GPU qx_upsample" tags = [:gpu] setup = [GPUBackends] begin
    using GPUEnv, Random
    Random.seed!(57)
    x = randn(Float32, 16, 16)
    for backend in GPUBackends.backends
        xg = to_gpu(backend, x)
        qd = qx_downsample(xg)
        qu = qx_upsample(qd, size(xg))
        @test size(qu) == size(xg)
        # upsample should reverse the downsample on even+even sites
        qu_cpu = qx_upsample(qx_downsample(x), size(x))
        @test maximum(abs, Array(qu) .- qu_cpu) < 1.0f-5
    end
end

@testitem "GPU nsdfb edge cases and error paths" tags = [:gpu] setup = [GPUBackends] begin
    using GPUEnv, Random
    Random.seed!(58)
    x = randn(Float32, 16, 16)
    p = ContourletParams(J = 1, L_array = [2])
    for backend in GPUBackends.fft_backends
        xg = to_gpu(backend, x)
        # l_levels == 0 → [copy(bandpass)]
        sbs0 = nsdfb_decompose(xg, 0, Q2345)
        @test length(sbs0) == 1
        @test maximum(abs, Array(sbs0[1]) .- x) < 1.0f-5
        # nsdfb_reconstruct n == 1 → copy
        rec1 = nsdfb_reconstruct([xg], Q2345)
        @test maximum(abs, Array(rec1) .- x) < 1.0f-5
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
    x = randn(Float32, 16, 16)
    # Haar-like non-ladder QuincunxFilterPair
    haar = QuincunxFilterPair([0.5 0.5], [1.0 1.0], (1, 1), (1, 2))
    for backend in GPUBackends.backends
        xg = to_gpu(backend, x)
        # :col direction — GPU kernel path
        sb0g, sb1g = qfb_decompose(xg, haar; dir = :col)
        sb0c, sb1c = qfb_decompose(x, haar; dir = :col)
        @test maximum(abs, Array(sb0g) .- sb0c) < 1.0f-5
        @test maximum(abs, Array(sb1g) .- sb1c) < 1.0f-5
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
        @test maximum(abs, Array(rec_row) .- x) < 1.0f-4
        # ladder filter throws
        @test_throws ArgumentError qfb_decompose(xg, Q2345)
        @test_throws ArgumentError qfb_reconstruct(sb0g, sb1g, Q2345)
    end
end

@testitem "GPU DFB with non-ladder filter (exercises shear kernels)" tags = [:gpu] setup = [GPUBackends] begin
    using GPUEnv, Random
    Random.seed!(73)
    x = randn(Float32, 16, 16)
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
            @test maximum(abs, Array(sbs[k]) .- sbs_cpu[k]) < 1.0f-5
        end
        rec = dfb_reconstruct(sbs, haar)
        @test !(rec isa Array)
        @test maximum(abs, Array(rec) .- x) < 1.0f-4
    end
end

@testitem "GPU conv2d_sep matches CPU on small band (OOB regression)" tags = [:gpu] setup = [GPUBackends] begin
    using GPUEnv, Random
    Random.seed!(202)
    # 3×3 and 4×4 bands with a length-9 filter: previously the CPU path read OOB (single-fold
    # reflection) while the GPU path clamped to the edge, so the two diverged.  Both are now
    # fixed to use the same triangle-wave fold.
    h9 = Float32[
        0.02674875741081, -0.016864118442875, -0.07822326652899,
        0.266864118442875, 0.60294901823636, 0.266864118442875,
        -0.07822326652899, -0.016864118442875, 0.02674875741081,
    ]
    for (nr, nc) in [(3, 3), (4, 4)]
        x = randn(Float32, nr, nc)
        for backend in GPUBackends.backends
            xg = to_gpu(backend, x)
            for bnd in (:symmetric, :periodic)
                dg = similar(xg)
                Contourlets.conv2d_sep!(dg, xg, h9, h9; boundary = bnd)
                dc = Contourlets.conv2d_sep(x, h9, h9; boundary = bnd)
                @test all(isfinite, Array(dg))
                @test maximum(abs, Array(dg) .- dc) < 1.0f-5
            end
        end
    end
end

@testitem "GPU TICT/STICT match CPU, reconstruct, stay on the device" tags = [:gpu] setup = [GPUBackends] begin
    using GPUEnv, Random
    Random.seed!(74)
    p = ContourletParams(J = 2, L_array = [3, 2])
    for x in (randn(Float32, 32, 32), complex.(randn(Float32, 32, 32), randn(Float32, 32, 32)))
        flat(c) = collect(Iterators.flatten(Iterators.flatten(c.subbands)))
        for (fwd, inv, mk, sim) in (
                (tict_forward, tict_inverse, make_tict_workspace, similar_tict_coefficients),
                (stict_forward, stict_inverse, make_stict_workspace, similar_stict_coefficients),
            )
            ccpu = fwd(x, p)
            for backend in GPUBackends.fft_backends
                xg = to_gpu(backend, x)
                cg = fwd(xg, p)
                @test !(cg.coarse isa Array)
                @test all(s -> !(s isa Array), flat(cg))
                @test maximum(abs, Array(cg.coarse) .- ccpu.coarse) < 1.0f-4
                @test all(maximum(abs, Array(a) .- b) < 1.0f-4 for (a, b) in zip(flat(cg), flat(ccpu)))
                @test maximum(abs, Array(inv(cg, p)) .- x) < 1.0f-4
                # Workspace path on the device.
                # Float64 params: the workspace promotes the data to Float64.
                ws = mk(xg, p)
                @test eltype(ws) == promote_type(Float64, eltype(x))
                cw = sim(p, size(x); Td = eltype(ws), M = typeof(similar(xg, eltype(ws))))
                out = similar(xg, eltype(ws))
                (fwd === tict_forward ? tict_forward! : stict_forward!)(cw, xg, p; workspace = ws)
                (fwd === tict_forward ? tict_inverse! : stict_inverse!)(out, cw, p; workspace = ws)
                @test maximum(abs, Array(out) .- x) < 1.0f-4
                # Coefficients move between host and device.
                @test Adapt.adapt(Array, cg).coarse isa Array
            end
        end
    end
end

@testitem "GPU make_nsct_workspace migrates filter caches to device" tags = [:gpu] setup = [GPUBackends] begin
    using GPUEnv, Random
    Random.seed!(75)
    # make_nsct_workspace(image, params) promotes the workspace element type up to
    # the filters' precision, never down — so Float32-precision filters are needed
    # here to get a Float32 (rather than Float64) workspace on a Float32 image.
    p = ContourletParams(
        J = 1, L_array = [2],
        lp_filters = FilterPair{Float32}(Float32.(CDF97.h), Float32.(CDF97.g)),
        dfb_filters = QuincunxFilterPair{Float32}(
            Float32.(Q2345.h_q), Float32.(Q2345.g_q),
            Q2345.c_h, Q2345.c_g, Float32.(Q2345.f_ladder)
        )
    )
    x = randn(Float32, 32, 32)
    for backend in GPUBackends.fft_backends
        xg = to_gpu(backend, x)
        # make_nsct_workspace(image, params) infers M from image → dispatches
        # the _adapt_bank GPU overload and moves the pyramid masks with `_like`
        ws = make_nsct_workspace(xg, p; prewarm = false)
        @test ws isa ContourletWorkspace
        # NSDFB equivalent filters should now be on the device
        @test !(ws.nsdfb_banks[1].ana[1] isa Array)
        # the pyramid masks should also be on the device
        @test !(ws.nsp_ana[1] isa Array)
        @test !(ws.nsp_syn[end] isa Array)
    end
end

@testitem "GPU Adapt.adapt_structure for CT/NSCT coefficients" tags = [:gpu] setup = [GPUBackends] begin
    using GPUEnv, Random
    const Adapt = GPUBackends.Adapt
    Random.seed!(60)
    x = randn(Float32, 32, 32)
    p = ContourletParams(J = 2, L_array = [2, 3])
    for backend in GPUBackends.backends
        xg = to_gpu(backend, x)
        # CT: move CPU coeffs to device and back via adapt_structure
        cc = ct_forward(x, p)
        cc_dev = Adapt.adapt(typeof(xg), cc)
        @test !(cc_dev.coarse isa Array)
        cc_host = Adapt.adapt(Array, cc_dev)
        @test cc_host.coarse isa Array
        @test maximum(abs, cc_host.coarse .- cc.coarse) < 1.0f-6
        # NSCT: same round-trip
        nc = nsct_forward(x, p)
        nc_dev = Adapt.adapt(typeof(xg), nc)
        @test !(nc_dev.coarse isa Array)
        nc_host = Adapt.adapt(Array, nc_dev)
        @test nc_host.coarse isa Array
        @test maximum(abs, nc_host.coarse .- nc.coarse) < 1.0f-6
    end
end

@testitem "GPU CT workspace path matches CPU" tags = [:gpu] setup = [GPUBackends] begin
    using GPUEnv, Random
    Random.seed!(42)
    # Float32 filters so the device workspace stays Float32 (Metal has no Float64).
    p = ContourletParams(
        J = 2, L_array = [2, 3],
        lp_filters = FilterPair{Float32}(Float32.(CDF97.h), Float32.(CDF97.g)),
        dfb_filters = QuincunxFilterPair{Float32}(
            Float32.(Q2345.h_q), Float32.(Q2345.g_q),
            Q2345.c_h, Q2345.c_g, Float32.(Q2345.f_ladder)
        )
    )
    x = randn(Float32, 64, 64)
    ccpu = ct_forward(x, p)
    for backend in GPUBackends.fft_backends
        xg = to_gpu(backend, x)
        DT = typeof(xg)
        ws = make_workspace(xg, p)               # DFB equivalent filters on the device
        @test !(ws.dfb_runners[1].bank.ana[1] isa Array)
        cg = ct_forward(xg, p; workspace = ws)   # GPU allocating forward with workspace
        @test cg.coarse isa DT
        @test maximum(abs, Array(cg.coarse) .- ccpu.coarse) < 1.0f-5
        for j in 1:2, k in eachindex(ccpu.subbands[j])
            @test maximum(abs, Array(cg.subbands[j][k]) .- ccpu.subbands[j][k]) < 1.0f-5
        end
        rec = ct_inverse(cg, p; workspace = ws)  # GPU allocating inverse with workspace
        @test rec isa DT
        @test maximum(abs, Array(rec) .- Array(ct_inverse(cg, p))) < 1.0f-5
        @test maximum(abs, Array(rec) .- x) < 1.0f-4
        # Generic `_allocate_zeros` path for a non-Array matrix type.
        z = Contourlets._allocate_zeros(DT, Float32, (3, 5))
        @test z isa DT
        @test size(z) == (3, 5)
        @test all(iszero, Array(z))
    end
end

@testitem "GPU NSCT workspace path matches CPU" tags = [:gpu] setup = [GPUBackends] begin
    using GPUEnv, Random
    Random.seed!(42)
    p = ContourletParams(
        J = 2, L_array = [2, 3],
        lp_filters = FilterPair{Float32}(Float32.(CDF97.h), Float32.(CDF97.g)),
        dfb_filters = QuincunxFilterPair{Float32}(
            Float32.(Q2345.h_q), Float32.(Q2345.g_q),
            Q2345.c_h, Q2345.c_g, Float32.(Q2345.f_ladder)
        )
    )
    x = randn(Float32, 32, 32)
    ncpu = nsct_forward(x, p)
    for backend in GPUBackends.fft_backends
        xg = to_gpu(backend, x)
        DT = typeof(xg)
        # The device workspace plans its FFTs on the device.
        ws = make_nsct_workspace(xg, p)
        @test ws.fft_plan_fwd !== nothing
        ng = nsct_forward(xg, p; workspace = ws)
        @test ng.coarse isa DT
        @test maximum(abs, Array(ng.coarse) .- ncpu.coarse) < 1.0f-5
        for j in 1:2, k in eachindex(ncpu.subbands[j])
            @test maximum(abs, Array(ng.subbands[j][k]) .- ncpu.subbands[j][k]) < 1.0f-5
        end
        rec = nsct_inverse(ng, p; workspace = ws)
        @test rec isa DT
        @test maximum(abs, Array(rec) .- Array(nsct_inverse(ng, p))) < 1.0f-5
        @test maximum(abs, Array(rec) .- x) < 1.0f-4
        # In-place variants reuse the same buffers.
        nsct_forward!(ng, xg, p; workspace = ws)
        @test maximum(abs, Array(ng.subbands[1][1]) .- ncpu.subbands[1][1]) < 1.0f-5
        nsct_inverse!(rec, ng, p; workspace = ws)
        @test maximum(abs, Array(rec) .- x) < 1.0f-4
    end
end

@testitem "GPU non-ladder QFB periodic boundary matches CPU (both directions)" tags = [:gpu] setup = [GPUBackends] begin
    using GPUEnv, Random
    Random.seed!(42)
    x = randn(Float32, 8, 8)
    # 5-tap filter centred at 3: the first/last kept sample reads one sample past
    # the edge, which the CPU kernel wraps periodically; the device kernels must
    # wrap the same way (they used to fold symmetrically / drop out-of-range taps).
    h5 = Float64[1 / 16 1 / 4 3 / 8 1 / 4 1 / 16]
    wide_qfp = QuincunxFilterPair(h5, h5, (1, 3), (1, 3))
    # Haar pair with c_h = 2: PR holds only with periodic wrap (see test_dfb.jl).
    qfp_wrap = QuincunxFilterPair(
        reshape([0.5, 0.5], 1, 2), reshape([1.0, 1.0], 1, 2), (1, 2), (1, 1)
    )
    for dir in (:row, :col), qfp in (wide_qfp, qfp_wrap)
        sb0c, sb1c = qfb_decompose(x, qfp; dir = dir)
        recc = qfb_reconstruct(sb0c, sb1c, qfp; dir = dir)
        for backend in GPUBackends.backends
            sb0g, sb1g = qfb_decompose(to_gpu(backend, x), qfp; dir = dir)
            @test size(sb0g) == size(sb0c)
            @test maximum(abs, Array(sb0g) .- sb0c) < 1.0f-5
            @test maximum(abs, Array(sb1g) .- sb1c) < 1.0f-5
            recg = qfb_reconstruct(sb0g, sb1g, qfp; dir = dir)
            @test maximum(abs, Array(recg) .- recc) < 1.0f-5
            qfp === qfp_wrap && @test maximum(abs, Array(recg) .- x) < 1.0f-5
        end
    end
end

@testitem "GPU _to_device/_ensure_gpu are no-ops on device arrays" tags = [:gpu] setup = [GPUBackends] begin
    using GPUEnv
    ext = Base.get_extension(Contourlets, :ContourletsGPUExt)
    @test ext !== nothing
    x = randn(Float32, 4, 4)
    for backend in GPUBackends.backends
        xg = to_gpu(backend, x)
        be = ext._gpu_backend(xg)
        @test ext._to_device(be, xg) === xg
        v = view(xg, :, 1:2)
        @test ext._to_device(be, v) === v
        @test ext._ensure_gpu(be, xg) === xg
        @test ext._ensure_gpu(be, v) === v
        # Host inputs are uploaded.
        m = ext._ensure_gpu(be, x)
        @test !(m isa Array)
        @test Array(m) == x
        vh = ext._ensure_gpu(be, x[:, 1])
        @test !(vh isa Array)
        @test Array(vh) == x[:, 1]
    end
end
