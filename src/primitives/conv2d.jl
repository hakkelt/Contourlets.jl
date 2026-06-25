# 2-D convolution with configurable boundary extension.
#
# Two backends are provided:
#   _conv2d_direct! — time-domain FIR convolution (good for small kernels)
#   _conv2d_fftw!   — overlap-add via FFTW (good for large kernels)
#
# The public entry points conv2d! / conv2d select the backend automatically:
# FFTW is preferred when the kernel area exceeds a fixed threshold.

using FFTW, LinearAlgebra

# ────────────────────────────────────────────────────────────────────────────────
# Boundary extension helpers (type-parameterised for zero overhead)
# ────────────────────────────────────────────────────────────────────────────────

# All boundary helpers take B as a type parameter so the compiler emits
# fully specialised code with no dispatch overhead per call.
# The symmetric helper uses an in-range fast path (skips mod for interior
# pixels) followed by a branch-free abs-fold for out-of-range indices.
@inline function _clamp_idx(i::Int, n::Int, ::Val{:symmetric})::Int
    1 <= i <= n && return i
    n == 1 && return 1
    p = 2 * (n - 1)
    m = mod(i - 1, p)
    return n - abs(m - (n - 1))
end

@inline function _clamp_idx(i::Int, n::Int, ::Val{:periodic})::Int
    return mod(i - 1, n) + 1
end

@inline function _clamp_idx(i::Int, n::Int, ::Val{:zero})::Int
    return i
end

@inline function _get(
        src::AbstractMatrix{T}, i::Int, j::Int, n1::Int, n2::Int,
        bv::Val{B}
    )::T where {T, B}
    ii = _clamp_idx(i, n1, bv)
    jj = _clamp_idx(j, n2, bv)
    (B === :zero && (ii < 1 || ii > n1 || jj < 1 || jj > n2)) && return zero(T)
    return @inbounds src[ii, jj]
end

# ────────────────────────────────────────────────────────────────────────────────
# Direct (time-domain) backend — type-stable inner kernel
# ────────────────────────────────────────────────────────────────────────────────

"""
    _conv2d_direct!(dst, src, kernel, origin; boundary=:symmetric)

In-place time-domain 2-D convolution.  `origin` is the (row, col) 1-based index
of the zero-lag tap in `kernel`.  Column-major traversal for cache efficiency.
"""
function _conv2d_direct!(
        dst::AbstractMatrix{Td},
        src::AbstractMatrix{Td},
        kernel::AbstractMatrix{Tf},
        origin::Tuple{Int, Int};
        boundary::Symbol = :symmetric
    ) where {Td, Tf}
    if boundary === :symmetric
        _conv2d_direct_impl!(dst, src, kernel, origin, Val(:symmetric))
    elseif boundary === :periodic
        _conv2d_direct_impl!(dst, src, kernel, origin, Val(:periodic))
    else
        _conv2d_direct_impl!(dst, src, kernel, origin, Val(:zero))
    end
    return dst
end

@inline function _conv2d_direct_impl!(
        dst::AbstractMatrix{Td},
        src::AbstractMatrix{Td},
        kernel::AbstractMatrix{Tf},
        origin::Tuple{Int, Int},
        bv::Val{B}
    ) where {Td, Tf, B}
    n1, n2 = size(src)
    kn1, kn2 = size(kernel)
    o1, o2 = origin

    @inbounds for j in 1:n2
        for i in 1:n1
            acc = zero(Td)
            for l in 1:kn2
                dl = l - o2
                jj = j - dl
                for k in 1:kn1
                    dk = k - o1
                    ii = i - dk
                    acc += kernel[k, l] * _get(src, ii, jj, n1, n2, bv)
                end
            end
            dst[i, j] = acc
        end
    end
    return dst
end

# ────────────────────────────────────────────────────────────────────────────────
# FFTW backend
# ────────────────────────────────────────────────────────────────────────────────

"""
    _build_fft_plan(src_size, kernel_size, T) -> (plan_fwd, plan_inv, padded_size)

Build the FFTW plans for FFT convolution of a `src_size` image with a
`kernel_size` kernel.

No plan is cached at the package level: FFTW already caches its `MEASURE` wisdom
(the expensive timing step) globally in the C library, so rebuilding a plan for a
size that has been planned before is cheap (sub-millisecond).  This FFT path is
only reached for large kernels through the public [`conv2d`](@ref); the contourlet
transforms use the separable direct backend ([`conv2d_sep!`](@ref)).
"""
function _build_fft_plan(
        src_size::Tuple{Int, Int},
        kernel_size::Tuple{Int, Int},
        ::Type{T},
        threading::ThreadingPolicy = Auto()
    ) where {T <: AbstractFloat}
    n1, n2 = src_size
    k1, k2 = kernel_size
    p1 = nextprod([2, 3, 5], n1 + k1 - 1)
    p2 = nextprod([2, 3, 5], n2 + k2 - 1)
    buf = zeros(T, p1, p2)
    threaded = _use_threading(threading, T)
    FFTW.set_num_threads(threaded ? Threads.nthreads() : 1)
    fwd = plan_rfft(buf; flags = FFTW.MEASURE)
    inv = plan_irfft(fwd * buf, p1; flags = FFTW.MEASURE)
    return fwd, inv, (p1, p2)
end

"""
    _conv2d_fftw!(dst, src, kernel, origin, ws_fftw; boundary=:symmetric)

In-place FFT-based 2-D convolution using pre-allocated workspace arrays
`ws_fftw = (pad_src, pad_ker, fft_src, fft_ker, fft_out, plan_fwd, plan_inv, padded_size)`.
"""
function _conv2d_fftw!(
        dst::AbstractMatrix{T},
        src::AbstractMatrix{T},
        kernel::AbstractMatrix{T},
        origin::Tuple{Int, Int},
        ws_fftw;
        boundary::Symbol = :symmetric
    ) where {T}
    pad_src, pad_ker, fft_src_buf, fft_ker_buf, fft_out_buf,
        plan_fwd, plan_inv, (p1, p2) = ws_fftw
    n1, n2 = size(src)
    kn1, kn2 = size(kernel)
    o1, o2 = origin
    bv = Val(boundary)

    # Fill the padded source.  The circular convolution reads extended samples on
    # all four sides: indices above n1 (reach o1−1 below the image) sit at rows
    # n1+1 … n1+o1−1, while negative indices (reach kn1−o1 above the image) wrap
    # to the END of the padded array (rows p1−(kn1−o1)+1 … p1).  Both regions are
    # filled with the requested boundary extension; the same applies to columns.
    ext_lo1, ext_hi1 = kn1 - o1, o1 - 1   # extension above / below the image
    ext_lo2, ext_hi2 = kn2 - o2, o2 - 1
    fill!(pad_src, zero(T))
    @inbounds for J in 1:p2
        # map padded column J to (possibly out-of-range) source column j
        j = J <= n2 + ext_hi2 ? J : J - p2
        (j > n2 + ext_hi2 || j < 1 - ext_lo2) && continue
        for I in 1:p1
            i = I <= n1 + ext_hi1 ? I : I - p1
            (i > n1 + ext_hi1 || i < 1 - ext_lo1) && continue
            pad_src[I, J] = _get(src, i, j, n1, n2, bv)
        end
    end

    # Place kernel in pad_ker with zero-phase wrap
    fill!(pad_ker, zero(T))
    @inbounds for l in 1:kn2, k in 1:kn1
        r = mod(k - o1, p1) + 1
        c = mod(l - o2, p2) + 1
        pad_ker[r, c] = kernel[k, l]
    end

    mul!(fft_src_buf, plan_fwd, pad_src)
    mul!(fft_ker_buf, plan_fwd, pad_ker)
    @inbounds for i in eachindex(fft_out_buf)
        fft_out_buf[i] = fft_src_buf[i] * fft_ker_buf[i]
    end
    mul!(pad_src, plan_inv, fft_out_buf)   # reuse pad_src as output buffer

    @inbounds for j in 1:n2, i in 1:n1
        dst[i, j] = pad_src[i, j]
    end
    return dst
end

# ────────────────────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────────────────────

"""
    conv2d!(dst, src, kernel, origin=(ceil(size(kernel,1)/2), ceil(size(kernel,2)/2));
            boundary=:symmetric) -> dst

In-place 2-D convolution of `src` with `kernel`, result written to `dst`.
`origin` is the 1-based (row, col) index of the zero-lag tap in `kernel`.
`boundary` ∈ {:symmetric, :periodic, :zero}.

Uses the FFTW backend when `length(kernel) > 25`, the direct backend otherwise.
"""
function conv2d!(
        dst::AbstractMatrix{Td},
        src::AbstractMatrix{Td},
        kernel::AbstractMatrix{Tf},
        origin::Tuple{Int, Int} = (
            ceil(Int, size(kernel, 1) / 2),
            ceil(Int, size(kernel, 2) / 2),
        );
        boundary::Symbol = :symmetric,
        threading::ThreadingPolicy = Auto()
    ) where {Td <: Number, Tf <: AbstractFloat}
    size(dst) == size(src) || throw(DimensionMismatch("dst and src must have the same size"))
    # The FFTW backend is real-only (plan_rfft) and needs data == kernel type, so
    # fall back to the direct backend for small kernels, complex data, or mixed
    # precision.
    if length(kernel) <= 25 || !(Td <: AbstractFloat && Td === Tf)
        _conv2d_direct!(dst, src, kernel, origin; boundary = boundary)
    else
        ws = _make_fftw_workspace(size(src), size(kernel), Td, threading)
        _conv2d_fftw!(dst, src, kernel, origin, ws; boundary = boundary)
    end
    return dst
end

"""
    conv2d(src, kernel, origin=(…); boundary=:symmetric) -> Matrix

Allocating wrapper around `conv2d!`.

!!! note
    For repeated calls with the same kernel size, preallocate a workspace with
    `_make_fftw_workspace` and call `conv2d!` directly — `plan_rfft(MEASURE)` is
    expensive (~14 ms for non-power-of-2 padded sizes) and is rebuilt on every
    call to this allocating form. The contourlet transforms never reach this path;
    they use `conv2d_sep!` exclusively.
"""
function conv2d(
        src::AbstractMatrix{Td},
        kernel::AbstractMatrix{Tf},
        origin::Tuple{Int, Int} = (
            ceil(Int, size(kernel, 1) / 2),
            ceil(Int, size(kernel, 2) / 2),
        );
        boundary::Symbol = :symmetric,
        threading::ThreadingPolicy = Auto()
    ) where {Td <: Number, Tf <: AbstractFloat}
    dst = similar(src)
    return conv2d!(dst, src, kernel, origin; boundary = boundary, threading = threading)
end

"""
    conv2d_sep!(dst, src, h_row, h_col; boundary=:symmetric, tmp=similar(src)) -> dst

Separable 2-D convolution: first apply 1-D filter `h_col` to each column, then
`h_row` to each row.  Both filters are symmetric (centre = mid-point).
Pass a preallocated `tmp` buffer (same size as `src`, distinct from `dst` and
`src`) to make the call allocation-free.

This is the workhorse for the Laplacian Pyramid stage.
"""
function conv2d_sep!(
        dst::AbstractMatrix{Td},
        src::AbstractMatrix{Td},
        h_row::AbstractVector{Tf},
        h_col::AbstractVector{Tf};
        boundary::Symbol = :symmetric,
        tmp::AbstractMatrix{Td} = similar(src)
    ) where {Td, Tf}
    if boundary === :symmetric
        _conv2d_sep_impl!(dst, src, h_row, h_col, tmp, Val(:symmetric))
    elseif boundary === :periodic
        _conv2d_sep_impl!(dst, src, h_row, h_col, tmp, Val(:periodic))
    else
        _conv2d_sep_impl!(dst, src, h_row, h_col, tmp, Val(:zero))
    end
    return dst
end

@inline function _conv2d_sep_impl!(
        dst::AbstractMatrix{Td},
        src::AbstractMatrix{Td},
        h_row::AbstractVector{Tf},
        h_col::AbstractVector{Tf},
        tmp::AbstractMatrix{Td},
        bv::Val{B}
    ) where {Td, Tf, B}
    n1, n2 = size(src)
    lr = length(h_row)
    lc = length(h_col)
    cr = (lr + 1) ÷ 2   # centre of row filter
    cc = (lc + 1) ÷ 2   # centre of col filter

    # Filter along columns (axis-1) first — accesses memory column-major.
    # `acc` carries the data type Td (real filter × Td data → Td).
    @inbounds for j in 1:n2
        for i in 1:n1
            acc = zero(Td)
            for k in 1:lc
                acc += h_col[k] * _get(src, i - (k - cc), j, n1, n2, bv)
            end
            tmp[i, j] = acc
        end
    end
    # Filter along rows (axis-2)
    @inbounds for j in 1:n2
        for i in 1:n1
            acc = zero(Td)
            for k in 1:lr
                acc += h_row[k] * _get(tmp, i, j - (k - cr), n1, n2, bv)
            end
            dst[i, j] = acc
        end
    end
    return dst
end

# Specialised fast-path for real CPU arrays with :symmetric or :periodic boundary.
# Interior pixels (where no boundary extension is needed) are split out and
# processed with @simd on unit-stride inner loops.  The thin border band falls
# back to the _get path.  We restructure to "zero then accumulate over k" so the
# innermost @simd loop is a simple axpy with stride-1 accesses — avoids the
# LoopVectorization gensym precompile bug triggered by 3-level @turbo reductions.
function _conv2d_sep_impl!(
        dst::AbstractMatrix{Td},
        src::AbstractMatrix{Td},
        h_row::AbstractVector{Tf},
        h_col::AbstractVector{Tf},
        tmp::AbstractMatrix{Td},
        bv::Union{Val{:symmetric}, Val{:periodic}}
    ) where {Td <: Real, Tf}
    n1, n2 = size(src)
    lr = length(h_row)
    lc = length(h_col)
    cr = (lr + 1) ÷ 2
    cc = (lc + 1) ÷ 2

    # Interior row range for the column pass.  When n1 < lc the range is empty.
    i_lo = cc
    i_hi = n1 - lc + cc

    # Column pass — border rows via _get
    for j in 1:n2
        @inbounds for i in 1:min(cc - 1, n1)
            acc = zero(Td)
            for k in 1:lc
                acc += h_col[k] * _get(src, i - (k - cc), j, n1, n2, bv)
            end
            tmp[i, j] = acc
        end
        @inbounds for i in max(n1 - lc + cc + 1, cc):n1
            acc = zero(Td)
            for k in 1:lc
                acc += h_col[k] * _get(src, i - (k - cc), j, n1, n2, bv)
            end
            tmp[i, j] = acc
        end
    end
    # Column pass — interior: zero then accumulate (k outer, @simd over i)
    @inbounds for j in 1:n2
        @simd for i in i_lo:i_hi
            tmp[i, j] = zero(Td)
        end
    end
    @inbounds for k in 1:lc
        c = h_col[k]
        row_off = cc - k
        for j in 1:n2
            @simd for i in i_lo:i_hi
                tmp[i, j] += c * src[i + row_off, j]
            end
        end
    end

    # Interior column range for the row pass.  When n2 < lr the range is empty.
    j_lo = cr
    j_hi = n2 - lr + cr

    # Row pass — border columns via _get
    @inbounds for j in 1:min(cr - 1, n2)
        for i in 1:n1
            acc = zero(Td)
            for k in 1:lr
                acc += h_row[k] * _get(tmp, i, j - (k - cr), n1, n2, bv)
            end
            dst[i, j] = acc
        end
    end
    @inbounds for j in max(n2 - lr + cr + 1, cr):n2
        for i in 1:n1
            acc = zero(Td)
            for k in 1:lr
                acc += h_row[k] * _get(tmp, i, j - (k - cr), n1, n2, bv)
            end
            dst[i, j] = acc
        end
    end
    # Row pass — interior: zero then accumulate (k outer, @simd over i)
    @inbounds for j in j_lo:j_hi
        @simd for i in 1:n1
            dst[i, j] = zero(Td)
        end
    end
    @inbounds for k in 1:lr
        c = h_row[k]
        col_off = cr - k
        @inbounds for j in j_lo:j_hi
            @simd for i in 1:n1
                dst[i, j] += c * tmp[i, j + col_off]
            end
        end
    end

    return dst
end

"""
    conv2d_sep(src, h_row, h_col; boundary=:symmetric) -> Matrix

Allocating version of `conv2d_sep!`.
"""
function conv2d_sep(
        src::AbstractMatrix{Td},
        h_row::AbstractVector{Tf},
        h_col::AbstractVector{Tf};
        boundary::Symbol = :symmetric
    ) where {Td, Tf}
    dst = similar(src)
    return conv2d_sep!(dst, src, h_row, h_col; boundary = boundary)
end

# ────────────────────────────────────────────────────────────────────────────────
# Internal FFT workspace builder — used only by `conv2d!` for large kernels.
# The contourlet transforms and `ContourletWorkspace` never reach this path
# (they convolve with small separable filters via `conv2d_sep!`).
# ────────────────────────────────────────────────────────────────────────────────

function _make_fftw_workspace(
        src_size::Tuple{Int, Int},
        kernel_size::Tuple{Int, Int},
        ::Type{T},
        threading::ThreadingPolicy = Auto()
    ) where {T <: AbstractFloat}
    fwd, inv, (p1, p2) = _build_fft_plan(src_size, kernel_size, T, threading)
    pad_src = zeros(T, p1, p2)
    pad_ker = zeros(T, p1, p2)
    Nc = p1 ÷ 2 + 1
    fft_src = zeros(Complex{T}, Nc, p2)
    fft_ker = zeros(Complex{T}, Nc, p2)
    fft_out = zeros(Complex{T}, Nc, p2)
    return (pad_src, pad_ker, fft_src, fft_ker, fft_out, fwd, inv, (p1, p2))
end
