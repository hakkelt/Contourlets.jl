# Rectangular-grid up- and down-sampling operators.
#
# Downsampling (↓2): keep every 2nd row AND every 2nd column (odd 1-based indices: 1, 3, 5, …).
# Upsampling   (↑2): insert a zero row and zero column after every kept sample.
#
# These operators are adjoint of each other (up to normalisation), which gives the
# LP stage its perfect-reconstruction property.

"""
    rect_downsample!(dst, src) -> dst

In-place rectangular 2×2 downsampling: `dst[i,j] = src[2i-1, 2j-1]`.
`dst` must have size `(ceil(n1/2), ceil(n2/2))`.

# Examples
```jldoctest
julia> using Contourlets

julia> x = reshape(1.0:16.0, 4, 4);

julia> rect_downsample(x)
2×2 Matrix{Float64}:
 1.0   9.0
 3.0  11.0
```
"""
function rect_downsample!(dst::AbstractMatrix{T}, src::AbstractMatrix{T}) where {T}
    @inbounds for j in axes(dst, 2), i in axes(dst, 1)
        dst[i, j] = src[2i - 1, 2j - 1]
    end
    return dst
end

"""
    rect_downsample(src) -> Matrix

Allocating rectangular downsample.
"""
function rect_downsample(src::AbstractMatrix{T}) where {T}
    n1, n2 = size(src)
    dst = similar(src, cld(n1, 2), cld(n2, 2))
    return rect_downsample!(dst, src)
end

"""
    rect_upsample!(dst, src) -> dst

In-place rectangular 2×2 upsampling: `dst[2i-1, 2j-1] = src[i,j]`, all other
entries zero.  `dst` must have size `(2*d1, 2*d2)` where `(d1,d2) = size(src)`.

# Examples
```jldoctest
julia> using Contourlets

julia> rect_upsample([1.0 2.0; 3.0 4.0])
4×4 Matrix{Float64}:
 1.0  0.0  2.0  0.0
 0.0  0.0  0.0  0.0
 3.0  0.0  4.0  0.0
 0.0  0.0  0.0  0.0
```
"""
function rect_upsample!(dst::AbstractMatrix{T}, src::AbstractMatrix{T}) where {T}
    fill!(dst, zero(T))
    @inbounds for j in axes(src, 2), i in axes(src, 1)
        dst[2i - 1, 2j - 1] = src[i, j]
    end
    return dst
end

"""
    rect_upsample(src) -> Matrix

Allocating rectangular upsample.
"""
function rect_upsample(src::AbstractMatrix{T}) where {T}
    d1, d2 = size(src)
    # `similar(src, …)` (not `zeros`) so the result matches `src`'s array type,
    # which keeps the allocating wrapper working for GPU arrays too.
    dst = similar(src, 2d1, 2d2)
    return rect_upsample!(dst, src)
end
