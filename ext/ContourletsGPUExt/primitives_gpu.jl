# GPU KernelAbstractions kernels for the Contourlet primitive operations.
#
# Each CPU primitive (conv2d_sep!, shear!, rect_downsample!, …) is overloaded
# for _AbstractGPUMatrix via a specialised @kernel that runs on any KA backend
# (CUDA, AMDGPU, oneAPI, Metal, CPU).

# ── Separable 2-D convolution ─────────────────────────────────────────────────

# Boundary remap of a (possibly out-of-range) 1-based index `p` into `1:n`.
# `bmode`: 1 = symmetric reflect, 2 = periodic.  Branch is uniform across the
# work-group, so it does not cause warp divergence in practice.
@inline function _gpu_boundary_idx(p::Int, n::Int, bmode::Int)
    if bmode == 2
        return mod(p - 1, n) + 1
    else
        1 <= p <= n && return p
        n == 1 && return 1
        period = 2 * (n - 1)
        m = mod(p - 1, period)
        return n - abs(m - (n - 1))
    end
end

# Pass 1: apply 1-D filter `h` along columns (axis-1), write to `tmp`.
@kernel function _conv_cols_kernel!(
        tmp, @Const(src), @Const(h),
        n1::Int, n2::Int, lc::Int, cc::Int, bmode::Int
    )
    i, j = @index(Global, NTuple)
    T = eltype(tmp)
    acc = zero(T)
    @inbounds for k in 1:lc
        ii = _gpu_boundary_idx(i - (k - cc), n1, bmode)
        acc += h[k] * src[ii, j]
    end
    @inbounds tmp[i, j] = acc
end

# Pass 2: apply 1-D filter `h` along rows (axis-2), write to `dst`.
@kernel function _conv_rows_kernel!(
        dst, @Const(tmp), @Const(h),
        n1::Int, n2::Int, lr::Int, cr::Int, bmode::Int
    )
    i, j = @index(Global, NTuple)
    T = eltype(dst)
    acc = zero(T)
    @inbounds for k in 1:lr
        jj = _gpu_boundary_idx(j - (k - cr), n2, bmode)
        acc += h[k] * tmp[i, jj]
    end
    @inbounds dst[i, j] = acc
end

"""
    conv2d_sep!(dst, src, h_row, h_col; boundary=:symmetric)

GPU-specialised separable convolution.  Dispatches via `_AbstractGPUMatrix`.
The filters `h_row` / `h_col` may be CPU or GPU vectors; they are moved to the
device automatically.  Supports `:symmetric` and `:periodic` boundaries.
"""
function conv2d_sep!(
        dst::_AbstractGPUMatrix{T},
        src::_AbstractGPUMatrix,
        h_row::AbstractVector,
        h_col::AbstractVector;
        boundary::Symbol = :symmetric,
        tmp = nothing
    ) where {T}
    bmode = if boundary === :symmetric
        1
    elseif boundary === :periodic
        2
    else
        throw(ArgumentError("GPU conv2d_sep! supports :symmetric or :periodic, got :$boundary"))
    end
    backend = _gpu_backend(src)
    n1, n2 = size(src)
    lc = length(h_col)
    lr = length(h_row)
    cc = (lc + 1) ÷ 2
    cr = (lr + 1) ÷ 2

    # Keep filters real (Tf); the kernels accumulate in the data type Td (= T)
    # so real·complex stays complex without promoting the filter.
    Tf = real(T)
    # Check for a task-local filter cache (used during CUDA graph capture to
    # avoid per-call pool allocations that would become graph memory nodes).
    fcache = Contourlets._active_filter_cache()
    function _get_filter_d(f::AbstractVector)
        raw = f isa AbstractVector{Tf} ? f : Tf.(f)
        (fcache !== nothing && haskey(fcache, f)) && return fcache[f]
        return _ensure_gpu(backend, raw)
    end
    h_col_d = _get_filter_d(h_col)
    h_row_d = _get_filter_d(h_row)

    _tmp = tmp === nothing ? _scratch_like(dst, n1, n2) : tmp

    kernel1 = _conv_cols_kernel!(backend, (16, 16))
    kernel1(_tmp, src, h_col_d, n1, n2, lc, cc, bmode; ndrange = (n1, n2))

    kernel2 = _conv_rows_kernel!(backend, (16, 16))
    kernel2(dst, _tmp, h_row_d, n1, n2, lr, cr, bmode; ndrange = (n1, n2))

    return dst
end

# ── Shearing ─────────────────────────────────────────────────────────────────

@kernel function _shear_h_kernel!(dst, @Const(src), n2::Int)
    i, j = @index(Global, NTuple)
    @inbounds dst[i, j] = src[i, mod1(j + i, n2)]
end

@kernel function _shear_v_kernel!(dst, @Const(src), n1::Int)
    i, j = @index(Global, NTuple)
    @inbounds dst[i, j] = src[mod1(i + j, n1), j]
end

@kernel function _inv_shear_h_kernel!(dst, @Const(src), n2::Int)
    i, j = @index(Global, NTuple)
    @inbounds dst[i, j] = src[i, mod1(j - i, n2)]
end

@kernel function _inv_shear_v_kernel!(dst, @Const(src), n1::Int)
    i, j = @index(Global, NTuple)
    @inbounds dst[i, j] = src[mod1(i - j, n1), j]
end

function shear!(
        dst::_AbstractGPUMatrix{T},
        src::_AbstractGPUMatrix{T},
        dir::Symbol
    ) where {T}
    n1, n2 = size(src)
    size(dst) == size(src) || throw(DimensionMismatch("dst and src must have the same size"))
    backend = _gpu_backend(src)
    if dir === :h
        kernel = _shear_h_kernel!(backend, (16, 16))
        kernel(dst, src, n2; ndrange = (n1, n2))
    elseif dir === :v
        kernel = _shear_v_kernel!(backend, (16, 16))
        kernel(dst, src, n1; ndrange = (n1, n2))
    else
        throw(ArgumentError("dir must be :h or :v, got :$dir"))
    end
    return dst
end

function inv_shear!(
        dst::_AbstractGPUMatrix{T},
        src::_AbstractGPUMatrix{T},
        dir::Symbol
    ) where {T}
    n1, n2 = size(src)
    size(dst) == size(src) || throw(DimensionMismatch("dst and src must have the same size"))
    backend = _gpu_backend(src)
    if dir === :h
        kernel = _inv_shear_h_kernel!(backend, (16, 16))
        kernel(dst, src, n2; ndrange = (n1, n2))
    elseif dir === :v
        kernel = _inv_shear_v_kernel!(backend, (16, 16))
        kernel(dst, src, n1; ndrange = (n1, n2))
    else
        throw(ArgumentError("dir must be :h or :v, got :$dir"))
    end
    return dst
end

# ── Rectangular up/downsampling ───────────────────────────────────────────────

@kernel function _rect_downsample_kernel!(dst, @Const(src))
    i, j = @index(Global, NTuple)
    @inbounds dst[i, j] = src[2i - 1, 2j - 1]
end

@kernel function _rect_upsample_kernel!(dst, @Const(src))
    i, j = @index(Global, NTuple)
    @inbounds dst[2i - 1, 2j - 1] = src[i, j]
end

function rect_downsample!(dst::_AbstractGPUMatrix{T}, src::_AbstractGPUMatrix{T}) where {T}
    d1, d2 = size(dst)
    backend = _gpu_backend(src)
    kernel = _rect_downsample_kernel!(backend, (16, 16))
    kernel(dst, src; ndrange = (d1, d2))
    return dst
end

function rect_upsample!(dst::_AbstractGPUMatrix{T}, src::_AbstractGPUMatrix{T}) where {T}
    d1, d2 = size(src)
    backend = _gpu_backend(src)
    fill!(dst, zero(T))   # sparse write — even rows/cols stay zero
    kernel = _rect_upsample_kernel!(backend, (16, 16))
    kernel(dst, src; ndrange = (d1, d2))
    return dst
end

# ── Quincunx up/downsampling ─────────────────────────────────────────────────

@kernel function _qx_downsample_kernel!(dst, @Const(src))
    i, k = @index(Global, NTuple)
    j = isodd(i) ? 2k - 1 : 2k
    @inbounds dst[i, k] = src[i, j]
end

@kernel function _qx_upsample_kernel!(dst, @Const(src))
    i, k = @index(Global, NTuple)
    j = isodd(i) ? 2k - 1 : 2k
    @inbounds dst[i, j] = src[i, k]
end

function qx_downsample!(dst::_AbstractGPUMatrix{T}, src::_AbstractGPUMatrix{T}) where {T}
    d1, d2 = size(dst)
    backend = _gpu_backend(src)
    kernel = _qx_downsample_kernel!(backend, (16, 16))
    kernel(dst, src; ndrange = (d1, d2))
    return dst
end

function qx_upsample!(dst::_AbstractGPUMatrix{T}, src::_AbstractGPUMatrix{T}) where {T}
    d1, d2 = size(src)
    backend = _gpu_backend(src)
    fill!(dst, zero(T))   # sparse write — off-quincunx positions stay zero
    kernel = _qx_upsample_kernel!(backend, (16, 16))
    kernel(dst, src; ndrange = (d1, d2))
    return dst
end

"""
    qx_downsample(src::_AbstractGPUMatrix) -> GPU Matrix
"""
function qx_downsample(src::_AbstractGPUMatrix{T}) where {T}
    n1, n2 = size(src)
    d1 = n1
    d2 = max(1, n2 ÷ 2)
    backend = _gpu_backend(src)
    dst = KernelAbstractions.allocate(backend, T, d1, d2)   # kernel covers all elements
    return qx_downsample!(dst, src)
end

"""
    qx_upsample(src::_AbstractGPUMatrix, out_size) -> GPU Matrix
"""
function qx_upsample(src::_AbstractGPUMatrix{T}, out_size::Tuple{Int, Int}) where {T}
    backend = _gpu_backend(src)
    dst = KernelAbstractions.allocate(backend, T, out_size...)   # qx_upsample! fills via fill!
    return qx_upsample!(dst, src)
end

# ── Quincunx Filter Bank (QFB) ────────────────────────────────────────────────

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
        jj = j - dc
        # symmetric boundary
        if jj < 1
            jj = 2 - jj
        end
        if jj > n2
            jj = 2 * n2 - jj
        end
        if jj < 1
            jj = 1
        end
        if jj > n2
            jj = n2
        end
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
            j_out = j_src + dc
            if 1 <= j_out <= n2
                out[i, j_out] += g[l] * s0 + (iseven(dc) ? g[l] : -g[l]) * s1
            end
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
        ii = i - dc
        if ii < 1
            ii = 2 - ii
        end
        if ii > n1
            ii = 2 * n1 - ii
        end
        if ii < 1
            ii = 1
        end
        if ii > n1
            ii = n1
        end
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
            i_out = i_src + dc
            if 1 <= i_out <= n1
                out[i_out, j] += g[l] * s0 + (iseven(dc) ? g[l] : -g[l]) * s1
            end
        end
    end
end

function qfb_decompose(image::_AbstractGPUMatrix, qfp::QuincunxFilterPair; dir::Symbol = :col)
    Contourlets.is_ladder(qfp) &&
        throw(ArgumentError("ladder-mode filter pairs (e.g. Q2345) are not yet supported on GPU"))
    T = promote_type(eltype(image), eltype(qfp))
    backend = _gpu_backend(image)
    img = T.(image)
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

function qfb_reconstruct(
        sb0::_AbstractGPUMatrix, sb1::_AbstractGPUMatrix,
        qfp::QuincunxFilterPair; dir::Symbol = :col
    )
    Contourlets.is_ladder(qfp) &&
        throw(ArgumentError("ladder-mode filter pairs (e.g. Q2345) are not yet supported on GPU"))
    T = promote_type(eltype(sb0), eltype(sb1), eltype(qfp))
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
