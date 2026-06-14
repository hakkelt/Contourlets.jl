# GPU port of the Nonsubsampled Directional Filter Bank (NSDFB).
#
# The NSDFB is a non-decimated, per-pixel directional convolution along a
# lattice direction (di, dj) with periodic (circular) boundaries.  Every output
# pixel is independent, so each `_nsqfb_*` stage maps directly onto a
# KernelAbstractions kernel.  The recursive tree (`_nsdfb_split` / `_nsdfb_merge`)
# is element-type generic, so providing device methods for `_nsqfb_decompose` /
# `_nsqfb_reconstruct` keeps the entire tree on the device — no per-stage host
# transfers.  Each thread reproduces the CPU per-pixel reduction order, so the
# device result matches the host path.

# ── Kernels ───────────────────────────────────────────────────────────────────

@kernel function _nsqfb_decompose_kernel!(
        sb0, sb1, @Const(img), @Const(h0), @Const(h1),
        c0::Int, c1::Int, K0::Int, K1::Int,
        di::Int, dj::Int, n1::Int, n2::Int
    )
    i, j = @index(Global, NTuple)
    T = eltype(sb0)
    acc0 = zero(T)
    @inbounds for m in 1:K0
        d = m - c0
        acc0 += h0[m] * img[mod1(i - di * d, n1), mod1(j - dj * d, n2)]
    end
    acc1 = zero(T)
    @inbounds for m in 1:K1
        d = m - c1
        acc1 += h1[m] * img[mod1(i - di * d, n1), mod1(j - dj * d, n2)]
    end
    @inbounds sb0[i, j] = acc0
    @inbounds sb1[i, j] = acc1
end

@kernel function _nsqfb_reconstruct_kernel!(
        out, @Const(sb0), @Const(sb1), @Const(g0), @Const(g1),
        cg0::Int, cg1::Int, K0::Int, K1::Int,
        di::Int, dj::Int, scale, n1::Int, n2::Int
    )
    i, j = @index(Global, NTuple)
    T = eltype(out)
    acc = zero(T)
    @inbounds for m in 1:K0
        d = m - cg0
        acc += g0[m] * sb0[mod1(i - di * d, n1), mod1(j - dj * d, n2)]
    end
    @inbounds for m in 1:K1
        d = m - cg1
        acc += g1[m] * sb1[mod1(i - di * d, n1), mod1(j - dj * d, n2)]
    end
    @inbounds out[i, j] = scale * acc
end

# ── Device `qup`: à trous filters moved to the device once per NSDFB call ─────

function _qup_to_device(backend, qup)
    return (
        h0 = _to_device(backend, qup.h0), c0 = qup.c0,
        h1 = _to_device(backend, qup.h1), c1 = qup.c1,
        g0 = _to_device(backend, qup.g0), cg0 = qup.cg0,
        g1 = _to_device(backend, qup.g1), cg1 = qup.cg1,
        scale = qup.scale,
    )
end

# ── Per-stage device methods (dispatched by the generic tree recursion) ───────

function Contourlets._nsqfb_decompose(image::AbstractGPUMatrix, qup, dir::Tuple{Int, Int})
    backend = _gpu_backend(image)
    T = eltype(image)
    di, dj = dir
    n1, n2 = size(image)
    sb0 = KernelAbstractions.allocate(backend, T, n1, n2)
    sb1 = KernelAbstractions.allocate(backend, T, n1, n2)
    kernel = _nsqfb_decompose_kernel!(backend, (16, 16))
    kernel(
        sb0, sb1, image, qup.h0, qup.h1, qup.c0, qup.c1,
        length(qup.h0), length(qup.h1), di, dj, n1, n2; ndrange = (n1, n2)
    )
    return sb0, sb1
end

function Contourlets._nsqfb_reconstruct(
        sb0::AbstractGPUMatrix, sb1::AbstractGPUMatrix, qup, dir::Tuple{Int, Int}
    )
    backend = _gpu_backend(sb0)
    T = eltype(sb0)
    di, dj = dir
    n1, n2 = size(sb0)
    out = KernelAbstractions.allocate(backend, T, n1, n2)
    kernel = _nsqfb_reconstruct_kernel!(backend, (16, 16))
    kernel(
        out, sb0, sb1, qup.g0, qup.g1, qup.cg0, qup.cg1,
        length(qup.g0), length(qup.g1), di, dj, T(qup.scale), n1, n2; ndrange = (n1, n2)
    )
    return out
end

# ── Public NSDFB entry points (device dispatch) ───────────────────────────────

function nsdfb_decompose(
        bandpass::AbstractGPUMatrix, l_levels::Int,
        qfp::QuincunxFilterPair, tree_level::Int
    )
    l_levels >= 0 || throw(ArgumentError("l_levels must be ≥ 0"))
    tree_level >= 1 || throw(ArgumentError("tree_level must be ≥ 1"))
    l_levels == 0 && return [copy(bandpass)]
    T = promote_type(eltype(bandpass), eltype(qfp))
    img = T === eltype(bandpass) ? bandpass : T.(bandpass)
    backend = _gpu_backend(bandpass)
    qup = _qup_to_device(backend, Contourlets._upsample_qfp_1d(qfp, 2^(tree_level - 1), T))
    return Contourlets._nsdfb_split(img, l_levels, 1, qup)
end

function nsdfb_reconstruct(
        subbands::Vector{<:AbstractGPUMatrix},
        qfp::QuincunxFilterPair, tree_level::Int
    )
    n = length(subbands)
    ispow2(n) || throw(ArgumentError("number of subbands must be a power of 2"))
    n == 1 && return copy(subbands[1])
    T = promote_type(eltype(subbands[1]), eltype(qfp))
    backend = _gpu_backend(subbands[1])
    qup = _qup_to_device(backend, Contourlets._upsample_qfp_1d(qfp, 2^(tree_level - 1), T))
    l_levels = round(Int, log2(n))
    return Contourlets._nsdfb_merge(subbands, l_levels, 1, qup)
end
