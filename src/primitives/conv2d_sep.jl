# Separable 2-D convolution (conv2d_sep! / conv2d_sep): a column pass then a row
# pass with 1-D filters, plus a SIMD fast path for real CPU arrays.  The boundary
# extension helper `_get` lives in `conv2d.jl`.

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
    # The tap loop reads `i + cc - k` for `k ∈ 1:lc`, i.e. `i - (lc - cc)` up to
    # `i + (cc - 1)`; for an even-length filter `lc - cc == cc`, one tap further
    # back than `cc`, so the lower bound is not `cc`.
    i_lo = lc - cc + 1
    i_hi = n1 - lc + cc

    # Column pass — border rows via _get
    for j in 1:n2
        @inbounds for i in 1:min(i_lo - 1, n1)
            acc = zero(Td)
            for k in 1:lc
                iszero(h_col[k]) || (acc += h_col[k] * _get(src, i - (k - cc), j, n1, n2, bv))
            end
            tmp[i, j] = acc
        end
        @inbounds for i in max(i_hi + 1, i_lo):n1
            acc = zero(Td)
            for k in 1:lc
                iszero(h_col[k]) || (acc += h_col[k] * _get(src, i - (k - cc), j, n1, n2, bv))
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
        iszero(c) && continue      # the holes of an à trous filter
        row_off = cc - k
        for j in 1:n2
            @simd for i in i_lo:i_hi
                tmp[i, j] += c * src[i + row_off, j]
            end
        end
    end

    # Interior column range for the row pass.  When n2 < lr the range is empty.
    # Same asymmetry as the column pass above for an even-length filter.
    j_lo = lr - cr + 1
    j_hi = n2 - lr + cr

    # Row pass — border columns via _get
    @inbounds for j in 1:min(j_lo - 1, n2)
        for i in 1:n1
            acc = zero(Td)
            for k in 1:lr
                iszero(h_row[k]) || (acc += h_row[k] * _get(tmp, i, j - (k - cr), n1, n2, bv))
            end
            dst[i, j] = acc
        end
    end
    @inbounds for j in max(j_hi + 1, j_lo):n2
        for i in 1:n1
            acc = zero(Td)
            for k in 1:lr
                iszero(h_row[k]) || (acc += h_row[k] * _get(tmp, i, j - (k - cr), n1, n2, bv))
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
        iszero(c) && continue
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
