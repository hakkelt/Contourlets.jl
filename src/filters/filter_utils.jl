# Filter utilities.

"""
    upsample_kernel(K::AbstractMatrix, factor::Integer) -> Matrix

2-D version of `MultiscaleFilterBanks.upsample_filter`: insert `factor - 1` zero rows and columns
between every row and column of kernel `K`.  `K` must be indexed from 1: taps are
placed by position.
"""
function upsample_kernel(K::AbstractMatrix{T}, factor::Integer)::Matrix{T} where {T}
    factor = Int(factor)
    factor >= 1 || throw(ArgumentError("factor must be ≥ 1"))
    Base.require_one_based_indexing(K)
    factor == 1 && return Matrix{T}(K)
    rows, cols = size(K)
    out = zeros(T, factor * (rows - 1) + 1, factor * (cols - 1) + 1)
    copyto!(view(out, 1:factor:size(out, 1), 1:factor:size(out, 2)), K)
    return out
end
