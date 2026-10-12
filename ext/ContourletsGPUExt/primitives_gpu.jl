# GPU KernelAbstractions kernels for the Contourlet primitive operations.
#
# Each CPU primitive (conv2d_sep!, shear!, rect_downsample!, …) is overloaded
# for _AbstractGPUMatrix via a specialised @kernel that runs on any KA backend
# (CUDA, AMDGPU, oneAPI, Metal, CPU).  The quincunx filter bank kernels are in
# `qfb_gpu.jl`.

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
function Contourlets.conv2d_sep!(
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
        throw(ArgumentError("GPU conv2d_sep! supports :symmetric or :periodic, got :$(boundary)"))
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

function Contourlets.shear!(
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
        throw(ArgumentError("dir must be :h or :v, got :$(dir)"))
    end
    return dst
end

function Contourlets.inv_shear!(
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
        throw(ArgumentError("dir must be :h or :v, got :$(dir)"))
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

function Contourlets.rect_downsample!(dst::_AbstractGPUMatrix{T}, src::_AbstractGPUMatrix{T}) where {T}
    d1, d2 = size(dst)
    backend = _gpu_backend(src)
    kernel = _rect_downsample_kernel!(backend, (16, 16))
    kernel(dst, src; ndrange = (d1, d2))
    return dst
end

function Contourlets.rect_upsample!(dst::_AbstractGPUMatrix{T}, src::_AbstractGPUMatrix{T}) where {T}
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

function Contourlets.qx_downsample!(dst::_AbstractGPUMatrix{T}, src::_AbstractGPUMatrix{T}) where {T}
    d1, d2 = size(dst)
    backend = _gpu_backend(src)
    kernel = _qx_downsample_kernel!(backend, (16, 16))
    kernel(dst, src; ndrange = (d1, d2))
    return dst
end

function Contourlets.qx_upsample!(dst::_AbstractGPUMatrix{T}, src::_AbstractGPUMatrix{T}) where {T}
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
function Contourlets.qx_downsample(src::_AbstractGPUMatrix{T}) where {T}
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
function Contourlets.qx_upsample(src::_AbstractGPUMatrix{T}, out_size::Tuple{Int, Int}) where {T}
    backend = _gpu_backend(src)
    dst = KernelAbstractions.allocate(backend, T, out_size...)   # qx_upsample! fills via fill!
    return qx_upsample!(dst, src)
end
