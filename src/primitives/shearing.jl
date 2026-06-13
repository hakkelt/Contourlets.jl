# Shearing operators used in the Directional Filter Bank.
#
# Shearing re-orders samples on a lattice without any arithmetic — it is a
# pure index remap that runs in O(N) time with zero multiplications.
#
# Two shear directions are supported:
#   :h (horizontal)  (i, j) → (i,  j + i  mod  n2)
#   :v (vertical)    (i, j) → (i + j  mod  n1,  j)
#
# The inverse of each shear is its mirror:
#   inv_shear!(:h)   (i, j) → (i,  j − i  mod  n2)
#   inv_shear!(:v)   (i, j) → (i − j  mod  n1,  j)
#
# Reference: Do & Vetterli (2005), Section V, "Simplification Using Shearing".

"""
    shear!(dst, src, dir::Symbol) -> dst

In-place shearing of `src` into `dst`.
- `dir = :h`: horizontal shear  `dst[i,j] = src[i, mod1(j+i, n2)]`
- `dir = :v`: vertical shear    `dst[i,j] = src[mod1(i+j, n1), j]`

# Examples
```jldoctest
julia> using Contourlets

julia> x = reshape(1.0:9.0, 3, 3);

julia> shear(x, :h)
3×3 Matrix{Float64}:
 4.0  7.0  1.0
 8.0  2.0  5.0
 3.0  6.0  9.0
```
"""
function shear!(
        dst::AbstractMatrix{T}, src::AbstractMatrix{T},
        dir::Symbol
    ) where {T}
    n1, n2 = size(src)
    size(dst) == size(src) || throw(DimensionMismatch("dst and src must have the same size"))
    if dir === :h
        @inbounds for j in 1:n2, i in 1:n1
            dst[i, j] = src[i, mod1(j + i, n2)]
        end
    elseif dir === :v
        @inbounds for j in 1:n2, i in 1:n1
            dst[i, j] = src[mod1(i + j, n1), j]
        end
    else
        throw(ArgumentError("dir must be :h or :v, got :$dir"))
    end
    return dst
end

"""
    shear(src, dir::Symbol) -> Matrix

Allocating version of `shear!`.
"""
shear(src::AbstractMatrix, dir::Symbol) = shear!(similar(src), src, dir)

"""
    inv_shear!(dst, src, dir::Symbol) -> dst

In-place inverse shearing (undoes `shear!`).
- `dir = :h`: `dst[i,j] = src[i, mod1(j−i, n2)]`
- `dir = :v`: `dst[i,j] = src[mod1(i−j, n1), j]`
"""
function inv_shear!(
        dst::AbstractMatrix{T}, src::AbstractMatrix{T},
        dir::Symbol
    ) where {T}
    n1, n2 = size(src)
    size(dst) == size(src) || throw(DimensionMismatch("dst and src must have the same size"))
    if dir === :h
        @inbounds for j in 1:n2, i in 1:n1
            dst[i, j] = src[i, mod1(j - i, n2)]
        end
    elseif dir === :v
        @inbounds for j in 1:n2, i in 1:n1
            dst[i, j] = src[mod1(i - j, n1), j]
        end
    else
        throw(ArgumentError("dir must be :h or :v, got :$dir"))
    end
    return dst
end

"""
    inv_shear(src, dir::Symbol) -> Matrix

Allocating version of `inv_shear!`.
"""
inv_shear(src::AbstractMatrix, dir::Symbol) = inv_shear!(similar(src), src, dir)
