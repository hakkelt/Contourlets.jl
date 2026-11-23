# Separable 2-D convolution (conv2d_sep! / conv2d_sep): a column pass then a row
# pass with 1-D filters, plus a SIMD fast path for real CPU arrays.

# The boundary extension: `clamp_idx` (from MultiscaleFilterBanks) takes the mode
# B as a type parameter, so each mode compiles to its own code with no dispatch
# per sample.
@inline function _get(
        src::AbstractMatrix{T}, i::Int, j::Int, n1::Int, n2::Int,
        bv::Val{B}
    )::T where {T, B}
    ii = clamp_idx(i, n1, bv)
    jj = clamp_idx(j, n2, bv)
    (B === :zero && (ii < 1 || ii > n1 || jj < 1 || jj > n2)) && return zero(T)
    return @inbounds src[ii, jj]
end

@noinline _throw_boundary(boundary) =
    throw(ArgumentError("boundary must be :symmetric, :periodic or :zero, got :$(boundary)"))

"""
    conv2d_sep!(dst, src, h_row, h_col; boundary=:symmetric, tmp=similar(src)) -> dst

Separable 2-D convolution: first apply 1-D filter `h_col` to each column, then
`h_row` to each row.  Both filters are symmetric (center = mid-point).
`boundary` ∈ {:symmetric, :periodic, :zero}; any other value throws an `ArgumentError`.
Pass a preallocated `tmp` buffer (same size as `src`, distinct from `dst` and
`src`) to make the call allocation-free.

This is the workhorse for the Laplacian Pyramid stage.  All arrays and filters must
be indexed from 1.
"""
function conv2d_sep!(
        dst::AbstractMatrix{Td},
        src::AbstractMatrix{Td},
        h_row::AbstractVector{Tf},
        h_col::AbstractVector{Tf};
        boundary::Symbol = :symmetric,
        tmp::AbstractMatrix{Td} = similar(src)
    ) where {Td, Tf}
    Base.require_one_based_indexing(src, h_row, h_col)
    _check_size(size(src), "dst" => dst, "tmp" => tmp)
    if boundary === :symmetric
        _conv2d_sep_impl!(dst, src, h_row, h_col, tmp, Val(:symmetric))
    elseif boundary === :periodic
        _conv2d_sep_impl!(dst, src, h_row, h_col, tmp, Val(:periodic))
    elseif boundary === :zero
        _conv2d_sep_impl!(dst, src, h_row, h_col, tmp, Val(:zero))
    else
        _throw_boundary(boundary)
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
    cr = (lr + 1) ÷ 2   # center of row filter
    cc = (lc + 1) ÷ 2   # center of col filter

    # Filter along columns (axis-1) first — accesses memory column-major.
    # `acc` carries the data type Td (real filter × Td data → Td).
    @inbounds for j in 1:n2
        for i in 1:n1
            acc = zero(Td)
            for k in 1:lc
                iszero(h_col[k]) || (acc += h_col[k] * _get(src, i - (k - cc), j, n1, n2, bv))
            end
            tmp[i, j] = acc
        end
    end
    # Filter along rows (axis-2)
    @inbounds for j in 1:n2
        for i in 1:n1
            acc = zero(Td)
            for k in 1:lr
                iszero(h_row[k]) || (acc += h_row[k] * _get(tmp, i, j - (k - cr), n1, n2, bv))
            end
            dst[i, j] = acc
        end
    end
    return dst
end

# Specialised fast-path for real CPU arrays with :symmetric or :periodic boundary.
# Interior pixels (where no boundary extension is needed) are split out and
# processed with @simd on unit-stride inner loops.  The thin border band falls
# back to the _get path.  The loops are ordered "zero then accumulate over k" so
# the innermost @simd loop is a simple axpy with stride-1 accesses.
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
    # The tap loop reads `i + cc - k` for `k ∈ 1:lc`, i.e. `i - (lc - cc)` up to
    # `i + (cc - 1)`; for an even-length filter `lc - cc == cc`, one tap further
    # back than `cc`, so the lower bound is not `cc`.
    i_lo = lc - cc + 1
    i_hi = n1 - lc + cc

    # Column pass — border rows via _get, then the interior
    for j in 1:n2
        _col_taps!(tmp, src, h_col, cc, 1:min(i_lo - 1, n1), j, bv)
        _col_taps!(tmp, src, h_col, cc, max(i_hi + 1, i_lo):n1, j, bv)
    end
    _col_interior!(tmp, src, h_col, cc, i_lo:i_hi)

    # Interior column range for the row pass.  When n2 < lr the range is empty.
    # Same asymmetry as the column pass above for an even-length filter.
    j_lo = lr - cr + 1
    j_hi = n2 - lr + cr

    # Row pass — border columns via _get, then the interior
    for j in 1:min(j_lo - 1, n2)
        _row_taps!(dst, tmp, h_row, cr, j, bv)
    end
    for j in max(j_hi + 1, j_lo):n2
        _row_taps!(dst, tmp, h_row, cr, j, bv)
    end
    _row_interior!(dst, tmp, h_row, cr, j_lo:j_hi)

    return dst
end

# tmp[i, j] for i in `rows`: the column filter (centered at `cc`) applied to
# column j of src, reading through the boundary extension.  `acc` carries the
# data type (real filter × Td data → Td).
@inline function _col_taps!(tmp::AbstractMatrix{Td}, src, h_col, cc, rows, j, bv) where {Td}
    n1, n2 = size(src)
    @inbounds for i in rows
        acc = zero(Td)
        for k in 1:length(h_col)
            iszero(h_col[k]) || (acc += h_col[k] * _get(src, i - (k - cc), j, n1, n2, bv))
        end
        tmp[i, j] = acc
    end
    return tmp
end

# dst[:, j]: the row filter (centered at `cr`) applied along the rows of tmp at
# column j, reading through the boundary extension.
@inline function _row_taps!(dst::AbstractMatrix{Td}, tmp, h_row, cr, j, bv) where {Td}
    n1, n2 = size(tmp)
    @inbounds for i in 1:n1
        acc = zero(Td)
        for k in 1:length(h_row)
            iszero(h_row[k]) || (acc += h_row[k] * _get(tmp, i, j - (k - cr), n1, n2, bv))
        end
        dst[i, j] = acc
    end
    return dst
end

# tmp[rows, :]: the column filter at rows whose taps all lie inside src — zero,
# then accumulate one tap at a time (k outer, @simd over i).
@inline function _col_interior!(tmp::AbstractMatrix{Td}, src, h_col, cc, rows) where {Td}
    n2 = size(src, 2)
    @inbounds for j in 1:n2
        @simd for i in rows
            tmp[i, j] = zero(Td)
        end
    end
    @inbounds for k in 1:length(h_col)
        c = h_col[k]
        iszero(c) && continue      # the holes of an à trous filter
        row_off = cc - k
        for j in 1:n2
            @simd for i in rows
                tmp[i, j] += c * src[i + row_off, j]
            end
        end
    end
    return tmp
end

# dst[:, cols]: the row filter at columns whose taps all lie inside tmp, in the
# same zero-then-accumulate order.
@inline function _row_interior!(dst::AbstractMatrix{Td}, tmp, h_row, cr, cols) where {Td}
    n1 = size(tmp, 1)
    @inbounds for j in cols
        @simd for i in 1:n1
            dst[i, j] = zero(Td)
        end
    end
    @inbounds for k in 1:length(h_row)
        c = h_row[k]
        iszero(c) && continue
        col_off = cr - k
        for j in cols
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
