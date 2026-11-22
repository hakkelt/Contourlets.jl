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
    using Contourlets: SubbandVector
    using Adapt
    export backends, fft_backends, Adapt

    # `test/runtests.jl` activates the GPU overlay before the runner starts, but a
    # test item can also be run on its own — an IDE test runner, or a language
    # server's test tooling — and that starts in the plain test environment where
    # no backend package is loadable and `gpu_backends()` comes back empty.
    # Activating here keeps the items self-contained either way; when the overlay
    # already exists (`persist = true` writes it to `test/gpu_env`) this only
    # loads it.
    isempty(gpu_backends(; include_cpu_backends = true)) &&
        GPUEnv.activate(persist = true, include_cpu_backends = true)

    # All backends to exercise (JLArrays always present).  GPUEnv supplies the
    # backend packages from its overlay environment, so the tests need no direct
    # GPUArrays / KernelAbstractions / JLArrays dependency.
    # All test data below is Float32 (Metal does not support Float64), so every
    # backend — including Metal — can run the same test bodies.
    const backends = gpu_backends(; include_cpu_backends = true)

    # The directional filter banks filter on the DFT grid, so the transforms need a
    # backend with an FFT (cuFFT, rocFFT, …, or FFTW on JLArrays' host memory);
    # OpenCL has none.
    const fft_backends = gpu_backends(; include_cpu_backends = true, supports_fftw = true)

    # Coefficients as a host array: the storage of a `SubbandVector` (`Array` of
    # the vector itself would index a device vector element by element).
    host(c) = Array(c isa SubbandVector ? parent(c) : c)
    export host
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
        # rectangular down/up-sampling
        dn = Contourlets.rect_downsample(xg)
        @test maximum(abs, Array(dn) .- Contourlets.rect_downsample(x)) < 1.0f-5
        up = Contourlets.rect_upsample(dn)
        @test maximum(abs, Array(up) .- Contourlets.rect_upsample(Array(dn))) < 1.0f-5
        # quincunx down/up-sampling
        qd = Contourlets.qx_downsample(xg)
        @test maximum(abs, Array(qd) .- Contourlets.qx_downsample(x)) < 1.0f-5
    end
end

@testitem "GPU Laplacian pyramid matches CPU + PR" tags = [:gpu] setup = [GPUBackends] begin
    using GPUEnv, Random
    Random.seed!(51)
    x = randn(Float32, 32, 32)
    cc, bpc = Contourlets.lp_decompose(x, CDF97)
    for backend in GPUBackends.backends
        xg = to_gpu(backend, x)
        cg, bpg = Contourlets.lp_decompose(xg, CDF97)
        @test maximum(abs, Array(cg) .- cc) < 1.0f-5
        @test maximum(abs, Array(bpg) .- bpc) < 1.0f-5
        rec = Contourlets.lp_reconstruct(cg, bpg, CDF97)
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
            cc, bpc = Contourlets.nsp_decompose(x, CDF97, lvl)
            cg, bpg = Contourlets.nsp_decompose(xg, CDF97, lvl)
            @test maximum(abs, Array(cg) .- cc) < 1.0f-5
            @test maximum(abs, Array(bpg) .- bpc) < 1.0f-5
            rec = Contourlets.nsp_reconstruct(cg, bpg, CDF97, lvl)
            @test maximum(abs, Array(rec) .- x) < 1.0f-4
        end
    end
end

@testitem "GPU CT matches CPU and reconstructs" tags = [:gpu] setup = [GPUBackends] begin
    using GPUEnv, Random
    Random.seed!(53)
    x = randn(Float32, 64, 64)
    t = CT(scales = 2, direction_levels = [3, 2])
    ccpu = analyze(t, x)
    for backend in GPUBackends.fft_backends
        xg = to_gpu(backend, x)
        cgpu = analyze(t, xg)
        # Coefficients stay on the device; bring them back to compare to the CPU path.
        @test !(parent(cgpu) isa Array)
        @test maximum(abs, Array(parent(cgpu)) .- parent(ccpu)) < 1.0f-5
        rec = synthesize(t, cgpu)
        @test !(rec isa Array)
        @test maximum(abs, Array(rec) .- x) < 1.0f-4
    end
end

@testitem "GPU NSCT matches CPU, reconstructs, shift-invariant" tags = [:gpu] setup = [GPUBackends] begin
    using GPUEnv, Random
    Random.seed!(54)
    x = randn(Float32, 32, 32)
    t = NSCT(scales = 2, direction_levels = [3, 2])
    ncpu = analyze(t, x)
    for backend in GPUBackends.fft_backends
        xg = to_gpu(backend, x)
        ngpu = analyze(t, xg)
        @test maximum(abs, Array(ngpu) .- ncpu) < 1.0f-5
        rec = synthesize(t, ngpu)
        @test maximum(abs, Array(rec) .- x) < 1.0f-4
        # Shift invariance on device.
        s = (3, 5)
        nshift = analyze(t, to_gpu(backend, circshift(x, s)))
        @test maximum(abs, Array(subband(t, nshift, 6)) .- circshift(Array(subband(t, ngpu, 6)), s)) < 1.0f-4
    end
end

@testitem "GPU complex CT/NSCT match CPU (real filters on device)" tags = [:gpu] setup = [GPUBackends] begin
    using GPUEnv, Random
    Random.seed!(55)
    z = complex.(randn(Float32, 32, 32), randn(Float32, 32, 32))
    for t in (CT(scales = 2, direction_levels = [3, 2]), NSCT(scales = 2, direction_levels = [3, 2]))
        ccpu = analyze(t, z)
        for backend in GPUBackends.fft_backends
            zg = to_gpu(backend, z)
            cgpu = analyze(t, zg)
            @test eltype(cgpu) <: Complex
            # Every subband, not only the lowpass.
            for i in 1:nsubbands(t, size(z))
                @test maximum(abs, Array(subband(t, cgpu, i)) .- subband(t, ccpu, i)) < 1.0f-4
            end
            @test maximum(abs, Array(synthesize(t, cgpu)) .- z) < 1.0f-4
        end
    end
end

@testitem "GPU coefficients stay on the device" tags = [:gpu] setup = [GPUBackends] begin
    using GPUEnv, Random
    Random.seed!(56)
    x = randn(Float32, 32, 32)
    for backend in GPUBackends.fft_backends
        xg = to_gpu(backend, x)
        DT = typeof(xg)
        for t in (CT(scales = 2, direction_levels = [3, 2]), NSCT(scales = 2, direction_levels = [3, 2]), TICT(scales = 2, direction_levels = [3, 2]), STICT(scales = 2, direction_levels = [3, 2]))
            c = analyze(t, xg)
            storage = c isa SubbandVector ? parent(c) : c
            @test !(storage isa Array)              # not transferred to host
            @test subband(t, c, 1) isa DT           # subbands are device matrices
            @test subband(t, c, 2) isa DT
            @test !(similar_coefficients(t, xg) isa Array)
            r = synthesize(t, c)
            @test r isa DT
            @test maximum(abs, Array(r) .- x) < 1.0f-4
        end
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
    end
end

@testitem "GPU qx_upsample" tags = [:gpu] setup = [GPUBackends] begin
    using GPUEnv, Random
    Random.seed!(57)
    x = randn(Float32, 16, 16)
    for backend in GPUBackends.backends
        xg = to_gpu(backend, x)
        qd = Contourlets.qx_downsample(xg)
        qu = Contourlets.qx_upsample(qd, size(xg))
        @test size(qu) == size(xg)
        # upsample should reverse the downsample on even+even sites
        qu_cpu = Contourlets.qx_upsample(Contourlets.qx_downsample(x), size(x))
        @test maximum(abs, Array(qu) .- qu_cpu) < 1.0f-5
    end
end

@testitem "GPU nsdfb edge cases and error paths" tags = [:gpu] setup = [GPUBackends] begin
    using GPUEnv, Random
    Random.seed!(58)
    x = randn(Float32, 16, 16)
    for backend in GPUBackends.fft_backends
        xg = to_gpu(backend, x)
        # l_levels == 0 → [copy(bandpass)]
        sbs0 = Contourlets.nsdfb_decompose(xg, Q2345, 0)
        @test length(sbs0) == 1
        @test maximum(abs, Array(sbs0[1]) .- x) < 1.0f-5
        # nsdfb_reconstruct n == 1 → copy
        rec1 = Contourlets.nsdfb_reconstruct([xg], Q2345)
        @test maximum(abs, Array(rec1) .- x) < 1.0f-5
        # error: l_levels < 0
        @test_throws ArgumentError Contourlets.nsdfb_decompose(xg, Q2345, -1)
        # error: non-power-of-2 subbands
        sbs2 = Contourlets.nsdfb_decompose(xg, Q2345, 2)
        @test_throws ArgumentError Contourlets.nsdfb_reconstruct(sbs2[1:3], Q2345)
    end
end

@testitem "GPU qfb_decompose/reconstruct reject device arrays" tags = [:gpu] setup = [GPUBackends] begin
    using GPUEnv
    x = randn(Float32, 16, 16)
    for backend in GPUBackends.backends
        xg = to_gpu(backend, x)
        sg = to_gpu(backend, zeros(Float32, 16, 8))
        @test_throws ArgumentError Contourlets.qfb_decompose(xg, Q2345)
        @test_throws ArgumentError Contourlets.qfb_reconstruct(sg, sg, Q2345)
    end
end

@testitem "GPU conv2d_sep matches CPU on small band (OOB regression)" tags = [:gpu] setup = [GPUBackends] begin
    using GPUEnv, Random
    Random.seed!(202)
    # 3×3 and 4×4 bands with a length-9 filter, longer than the band: the CPU and GPU
    # paths must both reflect with the same triangle-wave fold, not a single fold or
    # an edge clamp.
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
    for x in (randn(Float32, 32, 32), complex.(randn(Float32, 32, 32), randn(Float32, 32, 32)))
        for t in (TICT(scales = 2, direction_levels = [2, 3]), STICT(scales = 2, direction_levels = [2, 3]))
            ccpu = analyze(t, x)
            for backend in GPUBackends.fft_backends
                xg = to_gpu(backend, x)
                cg = analyze(t, xg)
                @test !((cg isa SubbandVector ? parent(cg) : cg) isa Array)
                @test maximum(abs, host(cg) .- host(ccpu)) < 1.0f-4
                @test maximum(abs, Array(synthesize(t, cg)) .- x) < 1.0f-4
                # Workspace path on the device, at the data's precision.
                ws = make_workspace(t, xg)
                @test eltype(ws) == eltype(x)
                cw = similar_coefficients(t, xg)
                out = similar(xg)
                analyze!(cw, t, xg; workspace = ws)
                synthesize!(out, t, cw; workspace = ws)
                @test maximum(abs, Array(out) .- x) < 1.0f-4
                # Coefficients move between host and device.
                @test host(Adapt.adapt(Array, cg)) == host(cg)
            end
        end
    end
end

@testitem "GPU NSCT workspace has its masks on the device" tags = [:gpu] setup = [GPUBackends] begin
    using GPUEnv, Random
    Random.seed!(75)
    t = NSCT(scales = 1, direction_levels = [2])
    x = randn(Float32, 32, 32)
    for backend in GPUBackends.fft_backends
        xg = to_gpu(backend, x)
        # The image form takes the array type from the image: the NSDFB filters
        # and pyramid masks move to the device.
        ws = make_workspace(t, xg; prewarm = false)
        @test ws isa NSCTWorkspace{Float32}
        @test !(ws.engine.banks[1].ana[1] isa Array)
        @test !(ws.engine.pyramid_ana[1] isa Array)
        @test !(ws.engine.coarse_syn isa Array)
    end
end

@testitem "GPU Adapt moves coefficients between host and device" tags = [:gpu] setup = [GPUBackends] begin
    using GPUEnv, Random
    const Adapt = GPUBackends.Adapt
    Random.seed!(60)
    x = randn(Float32, 32, 32)
    for backend in GPUBackends.backends
        xg = to_gpu(backend, x)
        for t in (CT(scales = 2, direction_levels = [3, 2]), NSCT(scales = 2, direction_levels = [3, 2]))
            cc = analyze(t, x)
            cc_dev = Adapt.adapt(typeof(xg).name.wrapper, cc)
            @test !((cc_dev isa SubbandVector ? parent(cc_dev) : cc_dev) isa Array)
            cc_host = Adapt.adapt(Array, cc_dev)
            @test typeof(cc_host) == typeof(cc)
            @test cc_host == cc
        end
    end
end

@testitem "GPU CT workspace path matches CPU" tags = [:gpu] setup = [GPUBackends] begin
    using GPUEnv, Random
    Random.seed!(42)
    t = CT(scales = 2, direction_levels = [3, 2])
    x = randn(Float32, 64, 64)
    ccpu = analyze(t, x)
    for backend in GPUBackends.fft_backends
        xg = to_gpu(backend, x)
        DT = typeof(xg)
        ws = make_workspace(t, xg)               # DFB equivalent filters on the device
        @test eltype(ws) == Float32
        @test !(ws.dfb_runners[1].bank.ana[1] isa Array)
        cg = analyze(t, xg; workspace = ws)      # allocating analysis with a workspace
        @test subband(t, cg, 1) isa DT
        @test maximum(abs, host(cg) .- host(ccpu)) < 1.0f-5
        rec = synthesize(t, cg; workspace = ws)  # allocating synthesis with a workspace
        @test rec isa DT
        @test maximum(abs, Array(rec) .- Array(synthesize(t, cg))) < 1.0f-5
        @test maximum(abs, Array(rec) .- x) < 1.0f-4
        analyze!(cg, t, xg; workspace = ws)
        @test maximum(abs, host(cg) .- host(ccpu)) < 1.0f-5
        synthesize!(rec, t, cg; workspace = ws)
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
    t = NSCT(scales = 2, direction_levels = [3, 2])
    x = randn(Float32, 32, 32)
    ncpu = analyze(t, x)
    for backend in GPUBackends.fft_backends
        xg = to_gpu(backend, x)
        # The device workspace plans its FFTs on the device.
        ws = make_workspace(t, xg)
        @test !(ws.engine.X isa Array)
        ng = analyze(t, xg; workspace = ws)
        @test ng isa typeof(similar(xg, (1, 1, 1)))
        @test maximum(abs, Array(ng) .- ncpu) < 1.0f-5
        rec = synthesize(t, ng; workspace = ws)
        @test rec isa typeof(xg)
        @test maximum(abs, Array(rec) .- Array(synthesize(t, ng))) < 1.0f-5
        @test maximum(abs, Array(rec) .- x) < 1.0f-4
        # In-place variants reuse the same buffers.
        analyze!(ng, t, xg; workspace = ws)
        @test maximum(abs, Array(ng) .- ncpu) < 1.0f-5
        synthesize!(rec, t, ng; workspace = ws)
        @test maximum(abs, Array(rec) .- x) < 1.0f-4
        # Host coefficients and a host image with a device workspace go through copies.
        ch = similar_coefficients(t, x)
        analyze!(ch, t, x; workspace = ws)
        @test maximum(abs, ch .- ncpu) < 1.0f-5
        @test maximum(abs, synthesize!(similar(x), t, ch; workspace = ws) .- x) < 1.0f-4
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

@testitem "GPU NSCT/TICT chunked analysis, stacked subbands, STICT replay" tags = [:gpu] setup = [GPUBackends] begin
    using GPUEnv, Random
    Random.seed!(11)
    close(a, b) = maximum(abs, host(a) .- host(b)) < 1.0f-4
    for x in (randn(Float32, 64, 64), complex.(randn(Float32, 64, 64), randn(Float32, 64, 64)))
        for t in (NSCT(scales = 2, direction_levels = [3, 2]), TICT(scales = 2, direction_levels = [3, 2]))
            ccpu = analyze(t, x)
            for backend in GPUBackends.fft_backends
                xg = to_gpu(backend, x)
                ws = make_workspace(t, xg)
                c = similar_coefficients(t, xg)
                # The subbands are the slices of the stacked array, device arrays of
                # its own kind, which a chunk's batched FFT writes in place.
                @test Contourlets.MultiscaleFilterBanks.mask_bank_storage(Contourlets._Stacked(c, ws.offsets)) isa typeof(c)
                # The analysis runs in chunks: every FFT backend has batched plans.
                @test ws.engine.batch !== nothing
                analyze!(c, t, xg; workspace = ws)
                @test close(c, ccpu)
                y = similar(xg)
                synthesize!(y, t, c; workspace = ws)
                @test close(y, x)
                # Coefficients moved to the device work as well.
                ca = Adapt.adapt(typeof(xg).name.wrapper, ccpu)
                @test close(synthesize!(y, t, ca; workspace = ws), x)
            end
        end
        # STICT: a captured graph (CUDA) replays for a new input into the same
        # coefficients, and alternating coefficient sets replay their own graphs.
        x2 = x .+ 1
        st = STICT(scales = 2, direction_levels = [3, 2])
        ref1, ref2 = analyze(st, x), analyze(st, x2)
        for backend in GPUBackends.fft_backends
            xg = to_gpu(backend, x)
            ws = make_workspace(st, xg)
            c1, c2 = similar_coefficients(st, xg), similar_coefficients(st, xg)
            xw = similar(xg)
            y = similar(xw)
            graphs = nothing
            for round in 1:2, (xi, ref, c) in ((x, ref1, c1), (x2, ref2, c1), (x, ref1, c2))
                copyto!(xw, xi)
                analyze!(c, st, xw; workspace = ws)
                @test close(c, ref)
                synthesize!(y, st, c; workspace = ws)
                @test close(y, xi)
                round == 1 && (graphs = Set(objectid(e[2]) for e in ws.graph_cache.entries))
            end
            # no capture in the second round (none at all off CUDA)
            @test length(ws.graph_cache.entries) == (backend.name === :CUDA ? 2 : 0)
            @test Set(objectid(e[2]) for e in ws.graph_cache.entries) == graphs
            # The allocating calls run eagerly: new arrays every call, nothing to replay.
            @test close(synthesize(st, analyze(st, xw; workspace = ws); workspace = ws), x)
            @test Set(objectid(e[2]) for e in ws.graph_cache.entries) == graphs
        end
    end
end

@testitem "GPU CT replay after a GC" tags = [:gpu] setup = [GPUBackends] begin
    using GPUEnv, Random
    # A captured CT graph (CUDA) reads device filters it does not keep alive: they
    # live with the cache entry, so a replay after a GC still reads them.
    t = CT(scales = 3, direction_levels = [3, 3, 2])
    x = randn(Xoshiro(3), Float32, 64, 64)
    ref = analyze(t, x)
    for backend in GPUBackends.fft_backends
        xg = to_gpu(backend, x)
        ws = make_workspace(t, xg)
        c = analyze(t, xg; workspace = ws)
        analyze!(c, t, xg; workspace = ws)
        GC.gc(true)
        # reuse freed device memory with NaNs
        junk = [fill!(similar(xg, k), NaN32) for k in (length(CDF97.h), length(CDF97.g), 16) for _ in 1:20]
        analyze!(c, t, xg; workspace = ws)
        @test maximum(abs, host(c) .- host(ref)) < 1.0f-4
    end
end

@testitem "GPU CT replay: Float64 filters, resized scratch" tags = [:gpu] setup = [GPUBackends] begin
    using GPUEnv, Random
    # Float64 replays read Float64 filters (not `cu`'s Float32), and a call that
    # resizes the scratch buffers a graph points at drops the cached graphs.
    t = CT(scales = 3, direction_levels = [3, 3, 2])
    x = randn(Xoshiro(4), 64, 64)
    ref = analyze(t, x)
    for backend in GPUBackends.fft_backends
        xg = to_gpu(backend, x)
        ws = make_workspace(t, xg)
        c = analyze(t, xg; workspace = ws)
        y = similar(xg)
        for _ in 1:2   # capture, then replay
            analyze!(c, t, xg; workspace = ws)
            @test maximum(abs, host(c) .- host(ref)) < 1.0e-12
            synthesize!(y, t, c; workspace = ws)
            @test maximum(abs, Array(y) .- x) < 1.0e-12
        end
        # The public calls reject another image size before running; the seam
        # below them resizes the scratch arena before its DFB lookup throws.
        r0 = ws.fwd_scratch.replaced
        @test_throws ArgumentError Contourlets._ct_forward_ws!(c, to_gpu(backend, randn(Xoshiro(5), 32, 32)), ws)
        @test ws.fwd_scratch.replaced > r0
        for _ in 1:2
            analyze!(c, t, xg; workspace = ws)
            @test maximum(abs, host(c) .- host(ref)) < 1.0e-12
        end
        @test length(ws.graph_cache.entries) == (backend.name === :CUDA ? 1 : 0)
    end
end

@testitem "GPU NSCT/TICT workspaces tuned on the device" tags = [:gpu] setup = [GPUBackends] begin
    using GPUEnv, Random
    Random.seed!(12)
    close(a, b) = maximum(abs, Array(a) .- b) < 1.0f-4
    for x in (randn(Float32, 64, 64), complex.(randn(Float32, 64, 64), randn(Float32, 64, 64)))
        for t in (NSCT(scales = 2, direction_levels = [3, 2]), TICT(scales = 2, direction_levels = [3, 2]))
            ccpu = analyze(t, x)
            for backend in GPUBackends.fft_backends
                xg = to_gpu(backend, x)
                # Without tuning, the synthesis is not chunked.
                @test make_workspace(t, xg).engine.tuning[2] === nothing
                ws = make_workspace(t, xg; tune = true)
                @test ws.engine.tuning[2] === Contourlets._ns_forward_batch_plan
                c = similar_coefficients(t, xg)
                analyze!(c, t, xg; workspace = ws)
                @test close(c, ccpu)
                y = similar(xg)
                synthesize!(y, t, c; workspace = ws)
                @test close(y, x)
            end
        end
    end
end

@testitem "GPU interface conformance (JLArrays)" tags = [:gpu] setup = [GPUBackends] begin
    using GPUEnv, Random
    using MultiscaleTransformsBase.Testing: test_transform
    rng = Xoshiro(13)
    for backend in GPUBackends.fft_backends, x in (randn(rng, Float32, 32, 32), randn(rng, ComplexF32, 32, 32))
        backend.name === :JLArray || continue
        xg = to_gpu(backend, x)
        for t in (CT(scales = 2, direction_levels = [3, 2]), NSCT(scales = 2, direction_levels = [3, 2]), TICT(scales = 2, direction_levels = [1, 2]), STICT(scales = 2, direction_levels = [2, 3]))
            test_transform(t, xg; rtol = 1.0f-4)
        end
    end
end

@testitem "GPU workspaces accept and ignore flags and tasks (JLArrays)" tags = [:gpu] setup = [GPUBackends] begin
    using GPUEnv, Random
    x = randn(Xoshiro(14), Float32, 32, 32)
    for backend in GPUBackends.fft_backends
        backend.name === :JLArray || continue
        xg = to_gpu(backend, x)
        for t in (CT(scales = 2, direction_levels = [3, 2]), NSCT(scales = 2, direction_levels = [3, 2]), TICT(scales = 2, direction_levels = [1, 2]), STICT(scales = 2, direction_levels = [2, 3]))
            ws0 = make_workspace(t, xg)
            ws = make_workspace(t, xg; flags = 64, tasks = 4, prewarm = false)     # 64: FFTW.ESTIMATE
            @test typeof(ws) === typeof(ws0)
            t isa Union{NSCT, TICT} && @test length(ws.engine.buffers) == 1
            @test Array(analyze(ws, xg)) == Array(analyze(ws0, xg))
        end
    end
end

@testitem "GPU CT and STICT workspaces stage host images" tags = [:gpu] setup = [GPUBackends] begin
    using GPUEnv, Random
    x = randn(Xoshiro(15), Float32, 32, 32)
    for backend in GPUBackends.fft_backends
        xg = to_gpu(backend, x)
        for t in (CT(scales = 2, direction_levels = [3, 2]), STICT(scales = 2, direction_levels = [2, 3]))
            ws = make_workspace(t, xg)
            cd = analyze(t, xg; workspace = ws)
            # A host image into the device workspace gives the same coefficients.
            ch = analyze!(similar(cd), t, x; workspace = ws)
            for i in 1:nsubbands(t, size(x))
                @test Array(subband(t, ch, i)) ≈ Array(subband(t, cd, i))
            end
            # Synthesis into a host image.
            y = zeros(Float32, 32, 32)
            @test synthesize!(y, t, cd; workspace = ws) === y
            @test maximum(abs, y .- x) < 1.0f-4
        end
    end
end
