# Quincunx lattice up- and down-sampling.
#
# The quincunx decimation matrix used throughout this package is
#
#   M_q = [[1, 1], [-1, 1]]   |det M_q| = 2
#
# The even-sum sublattice (n₁+n₂ even) contains exactly n2÷2 columns per row:
#   odd rows  (i odd):  columns j = 1, 3, 5, … (odd+odd = even)
#   even rows (i even): columns j = 2, 4, 6, … (even+even = even)
#
# Row-parity polyphase packing maps these to a compact n1×(n2÷2) array:
#   k-th retained sample in row i → dst[i, k],  where j = isodd(i) ? 2k-1 : 2k
#
# This is the bijection used by qx_downsample / qx_upsample.  n2 must be even.

"""
    qx_downsample!(dst, src) -> dst

In-place quincunx downsampling using row-parity polyphase packing.

Retains the even-sum sublattice samples (`n₁+n₂` even) and packs them into
`dst` of size `(n1, n2÷2)`.  For each row `i`, the retained columns are
`j = 1,3,5,…` (odd rows) or `j = 2,4,6,…` (even rows); the `k`-th retained
column maps to `dst[i, k]`.

`n2` must be even; throws `ArgumentError` otherwise.

# Examples
```jldoctest
julia> using Contourlets

julia> x = reshape(1.0:16.0, 4, 4);

julia> size(qx_downsample(x))
(4, 2)
```
"""
function qx_downsample!(dst::AbstractMatrix{T}, src::AbstractMatrix{T}) where {T}
    n1, n2 = size(src)
    iseven(n2) || throw(ArgumentError("qx_downsample requires even number of columns, got n2=$n2"))
    fill!(dst, zero(T))
    d2 = size(dst, 2)
    @inbounds for k in 1:d2, i in 1:n1
        j = isodd(i) ? 2k - 1 : 2k
        dst[i, k] = src[i, j]
    end
    return dst
end

"""
    qx_downsample(src) -> Matrix
"""
function qx_downsample(src::AbstractMatrix{T}) where {T}
    n1, n2 = size(src)
    iseven(n2) || throw(ArgumentError("qx_downsample requires even number of columns, got n2=$n2"))
    dst = zeros(T, n1, n2 ÷ 2)
    return qx_downsample!(dst, src)
end

"""
    qx_upsample!(dst, src) -> dst

In-place quincunx upsampling: exact inverse of `qx_downsample!`.

Scatters `src[i, k]` back to `dst[i, j]` where `j = isodd(i) ? 2k-1 : 2k`;
all other entries of `dst` are set to zero.
"""
function qx_upsample!(dst::AbstractMatrix{T}, src::AbstractMatrix{T}) where {T}
    fill!(dst, zero(T))
    d1, d2 = size(src)
    @inbounds for k in 1:d2, i in 1:d1
        j = isodd(i) ? 2k - 1 : 2k
        dst[i, j] = src[i, k]
    end
    return dst
end

"""
    qx_upsample(src, out_size) -> Matrix
"""
function qx_upsample(src::AbstractMatrix{T}, out_size::Tuple{Int, Int}) where {T}
    dst = zeros(T, out_size...)
    return qx_upsample!(dst, src)
end
