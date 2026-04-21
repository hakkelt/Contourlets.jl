# Quincunx lattice up- and down-sampling.
#
# The quincunx decimation matrix used throughout this package is
#
#   M_q = [[1, 1], [-1, 1]]   |det M_q| = 2
#
# The M_q-lattice ("quincunx sublattice") consists of all integer points
# (n₁, n₂) with n₁ + n₂ even.  After downsampling by M_q the retained
# samples form a rectangular grid via the change of coordinates
#
#   (k₁, k₂) = M_q⁻¹ (n₁, n₂) = ((n₁−n₂)/2, (n₁+n₂)/2)
#
# i.e. we keep (n₁, n₂) whenever n₁+n₂ is even and map
#   k₁ = (n₁ − n₂)/2,  k₂ = (n₁ + n₂)/2
#
# The inverse (upsampling) inserts zeros on the odd-sum sublattice.

"""
    qx_downsample!(dst, src) -> dst

In-place quincunx downsampling.  Retains samples at positions where `n₁+n₂` is even.
The retained samples are written to `dst` in the coordinate frame of M_q⁻¹.
`dst` must have size `(cld(n1+n2,2), cld(n1+n2,2))` ≈ `(n1+n2)/2 × (n1+n2)/2`.

For an `n1×n2` image with n1=n2=N (power-of-2), `dst` has size `(N/2 + N/2, N/2)` —
concretely `(N, N/2)` after the diagonal coordinate change.

For simplicity this implementation maps (n₁,n₂) with n₁+n₂ even to
`dst[(n₁-n₂)/2 + offset₁, (n₁+n₂)/2 + offset₂]` with offsets chosen to keep
indices positive.

# Examples
```jldoctest
julia> using Contourlets
julia> x = reshape(1.0:16.0, 4, 4)
julia> y = qx_downsample(x); size(y)
(4, 2)
```
"""
function qx_downsample!(dst::AbstractMatrix{T}, src::AbstractMatrix{T}) where {T}
    n1, n2 = size(src)
    fill!(dst, zero(T))
    d1, d2 = size(dst)
    @inbounds for j in 1:n2, i in 1:n1
        if iseven(i + j)       # quincunx sublattice
            # coordinate change:  k1 = (i-j)/2 + 1,  k2 = (i+j)/2
            k1 = (i - j) ÷ 2 + (n2 + 1) ÷ 2   # shift to keep 1-based
            k2 = (i + j) ÷ 2
            if 1 <= k1 <= d1 && 1 <= k2 <= d2
                dst[k1, k2] = src[i, j]
            end
        end
    end
    return dst
end

"""
    qx_downsample(src) -> Matrix
"""
function qx_downsample(src::AbstractMatrix{T}) where {T}
    n1, n2 = size(src)
    d1 = n1   # after the skew coordinate change the size becomes n1 × n2/2 (approx)
    d2 = max(1, n2 ÷ 2)
    dst = zeros(T, d1, d2)
    return qx_downsample!(dst, src)
end

"""
    qx_upsample!(dst, src) -> dst

In-place quincunx upsampling: inverse of `qx_downsample!`.
Places `src[k₁,k₂]` back at position `(n₁,n₂)` in `dst` with the even-sum
constraint; all other entries are set to zero.
"""
function qx_upsample!(dst::AbstractMatrix{T}, src::AbstractMatrix{T}) where {T}
    fill!(dst, zero(T))
    n1, n2 = size(dst)
    d1, d2 = size(src)
    @inbounds for k2 in 1:d2, k1 in 1:d1
        i = k1 + k2 - (n2 + 1) ÷ 2   # inverse of k1 = (i-j)/2 + offset
        j = k2 * 2 - i               # from k2 = (i+j)/2
        if 1 <= i <= n1 && 1 <= j <= n2
            dst[i, j] = src[k1, k2]
        end
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
