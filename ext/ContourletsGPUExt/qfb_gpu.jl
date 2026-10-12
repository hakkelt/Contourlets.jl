# GPU KernelAbstractions kernels for the two-channel quincunx filter bank (QFB),
# and the qfb_decompose / qfb_reconstruct overloads for _AbstractGPUMatrix
# (modulation-mode filter pairs only).  Mirrors `src/directional/quincunx_fb.jl`.

# ── Quincunx Filter Bank (QFB) kernels ────────────────────────────────────────

# GPU kernel: QFB col-direction analysis.
# Each thread handles one (i, k2) pair where k2 is the output column index.
@kernel function _qfb_col_decompose_kernel_gpu!(
        sb0, sb1, @Const(img),
        @Const(h), c_h::Int, n1::Int, n2::Int, d2::Int, K::Int
    )
    i, k2 = @index(Global, NTuple)
    T = eltype(sb0)
    j = 2 * k2   # even column (1-indexed: j = 2, 4, 6, …)
    lp_val = zero(T)
    hp_val = zero(T)
    @inbounds for l in 1:K
        dc = l - c_h
        jj = mod1(j - dc, n2)   # periodic boundary, as the CPU kernel
        xval = img[i, jj]
        lp_val += h[l] * xval
        hp_val += (iseven(dc) ? h[l] : -h[l]) * xval
    end
    sb0[i, k2] = lp_val
    sb1[i, k2] = hp_val
end

# GPU kernel: QFB col-direction synthesis (row-parallel).
# One thread per row i; loops over all output columns (k2) serially.
# This avoids write-after-write races while still parallelising over rows.
@kernel function _qfb_col_reconstruct_kernel_gpu!(
        out, @Const(sb0), @Const(sb1),
        @Const(g), c_g::Int, n2::Int, d2::Int, K::Int
    )
    i = @index(Global)
    T = eltype(out)
    @inbounds for k2 in 1:d2
        j_src = 2 * k2
        s0 = sb0[i, k2]
        s1 = sb1[i, k2]
        for l in 1:K
            dc = l - c_g
            j_out = mod1(j_src + dc, n2)   # periodic scatter, as the CPU kernel
            out[i, j_out] += g[l] * s0 + (iseven(dc) ? g[l] : -g[l]) * s1
        end
    end
end

# GPU kernel: QFB row-direction analysis (mirrors col kernel with axes swapped).
# Each thread handles one (k1, j) pair where k1 is the output row index.
@kernel function _qfb_row_decompose_kernel_gpu!(
        sb0, sb1, @Const(img),
        @Const(h), c_h::Int, n1::Int, n2::Int, d1::Int, K::Int
    )
    k1, j = @index(Global, NTuple)
    T = eltype(sb0)
    i = 2 * k1   # even row (1-indexed: i = 2, 4, 6, …)
    lp_val = zero(T)
    hp_val = zero(T)
    @inbounds for l in 1:K
        dc = l - c_h
        ii = mod1(i - dc, n1)   # periodic boundary, as the CPU kernel
        xval = img[ii, j]
        lp_val += h[l] * xval
        hp_val += (iseven(dc) ? h[l] : -h[l]) * xval
    end
    sb0[k1, j] = lp_val
    sb1[k1, j] = hp_val
end

# GPU kernel: QFB row-direction synthesis (column-parallel, mirrors col-reconstruct).
# One thread per column j; loops over all output rows (k1) serially to avoid races.
@kernel function _qfb_row_reconstruct_kernel_gpu!(
        out, @Const(sb0), @Const(sb1),
        @Const(g), c_g::Int, n1::Int, d1::Int, K::Int
    )
    j = @index(Global)
    T = eltype(out)
    @inbounds for k1 in 1:d1
        i_src = 2 * k1
        s0 = sb0[k1, j]
        s1 = sb1[k1, j]
        for l in 1:K
            dc = l - c_g
            i_out = mod1(i_src + dc, n1)   # periodic scatter, as the CPU kernel
            out[i_out, j] += g[l] * s0 + (iseven(dc) ? g[l] : -g[l]) * s1
        end
    end
end

# ── Quincunx Filter Bank (QFB) ────────────────────────────────────────────────

function Contourlets.qfb_decompose(image::_AbstractGPUMatrix, qfp::QuincunxFilterPair; dir::Symbol = :col)
    Contourlets.is_ladder(qfp) &&
        throw(ArgumentError("ladder-mode filter pairs (e.g. Q2345) are not yet supported on GPU"))
    # Data eltype is authoritative (mirrors the CPU qfb_decompose): the filter is
    # converted to match, never promoted, so e.g. a Float32 device image stays
    # Float32 even against the default Float64 filter constants — Metal (which
    # has no Float64 support at all) depends on this.
    T = eltype(image)
    backend = _gpu_backend(image)
    img = image
    h_vec = vec(qfp.h_q)
    h_d = _ensure_gpu(backend, eltype(h_vec) === T ? h_vec : T.(h_vec))
    n1, n2 = size(img)
    K = length(h_d)
    if dir === :col
        c_h = qfp.c_h[2]
        d2 = n2 ÷ 2
        sb0 = KernelAbstractions.allocate(backend, T, n1, d2)   # kernel covers all elements
        sb1 = KernelAbstractions.allocate(backend, T, n1, d2)
        kernel = _qfb_col_decompose_kernel_gpu!(backend, (16, 16))
        kernel(sb0, sb1, img, h_d, c_h, n1, n2, d2, K; ndrange = (n1, d2))
        return sb0, sb1
    else
        c_h = qfp.c_h[2]
        d1 = n1 ÷ 2
        sb0 = KernelAbstractions.allocate(backend, T, d1, n2)   # kernel covers all elements
        sb1 = KernelAbstractions.allocate(backend, T, d1, n2)
        kernel = _qfb_row_decompose_kernel_gpu!(backend, (16, 16))
        kernel(sb0, sb1, img, h_d, c_h, n1, n2, d1, K; ndrange = (d1, n2))
        return sb0, sb1
    end
end

function Contourlets.qfb_reconstruct(
        sb0::_AbstractGPUMatrix, sb1::_AbstractGPUMatrix,
        qfp::QuincunxFilterPair; dir::Symbol = :col
    )
    Contourlets.is_ladder(qfp) &&
        throw(ArgumentError("ladder-mode filter pairs (e.g. Q2345) are not yet supported on GPU"))
    # Data eltype is authoritative (mirrors the CPU qfb_reconstruct) — see qfb_decompose.
    T = promote_type(eltype(sb0), eltype(sb1))
    backend = _gpu_backend(sb0)
    g_vec = vec(qfp.g_q)
    g_d = _ensure_gpu(backend, eltype(g_vec) === T ? g_vec : T.(g_vec))
    c_g = qfp.c_g[2]
    K = length(g_d)
    sb0_T = eltype(sb0) === T ? sb0 : T.(sb0)
    sb1_T = eltype(sb1) === T ? sb1 : T.(sb1)
    if dir === :col
        n1 = size(sb0, 1)
        n2 = size(sb0, 2) * 2
        d2 = size(sb0, 2)
        out = KernelAbstractions.zeros(backend, T, n1, n2)   # += accumulation needs zero init
        kernel = _qfb_col_reconstruct_kernel_gpu!(backend, 256)
        kernel(out, sb0_T, sb1_T, g_d, c_g, n2, d2, K; ndrange = n1)
        return out
    else
        d1 = size(sb0, 1)
        n1 = d1 * 2
        n2 = size(sb0, 2)
        out = KernelAbstractions.zeros(backend, T, n1, n2)   # += accumulation needs zero init
        kernel = _qfb_row_reconstruct_kernel_gpu!(backend, 256)
        kernel(out, sb0_T, sb1_T, g_d, c_g, n1, d1, K; ndrange = n2)
        return out
    end
end
