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
        # Forward result is host-side and must be bit-identical to the CPU path.
        @test maximum(abs, cgpu.coarse .- ccpu.coarse) < 1.0e-12
        for j in 1:2, k in eachindex(ccpu.subbands[j])
            @test maximum(abs, cgpu.subbands[j][k] .- ccpu.subbands[j][k]) < 1.0e-12
        end
        rec = ct_inverse(cgpu, xg)
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
        @test maximum(abs, ngpu.coarse .- ncpu.coarse) < 1.0e-12
        for j in 1:2, k in eachindex(ncpu.subbands[j])
            @test maximum(abs, ngpu.subbands[j][k] .- ncpu.subbands[j][k]) < 1.0e-12
        end
        rec = nsct_inverse(ngpu, xg)
        @test maximum(abs, Array(rec) .- x) < 1.0e-11
        # Shift invariance on device.
        s = (3, 5)
        nshift = nsct_forward(to_gpu(backend, circshift(x, s)), p)
        @test maximum(abs, nshift.subbands[2][1] .- circshift(ngpu.subbands[2][1], s)) < 1.0e-11
    end
end
