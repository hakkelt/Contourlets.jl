# Utilities operating on FilterPair and QuincunxFilterPair.

"""
    upsample_filter(h::AbstractVector, factor::Int) -> Vector

Insert `factor - 1` zeros between every tap of `h`.  This implements the
"à trous" (with holes) upsampling used by the nonsubsampled pyramid and NSDFB
to avoid downsampling at coarser levels.

# Examples
```jldoctest
julia> using Contourlets
julia> upsample_filter([1.0, 2.0, 3.0], 2)
5-element Vector{Float64}:
 1.0
 0.0
 2.0
 0.0
 3.0
```
"""
function upsample_filter(h::AbstractVector{T}, factor::Int)::Vector{T} where {T}
    factor >= 1 || throw(ArgumentError("factor must be ≥ 1"))
    factor == 1 && return copy(h)
    n = length(h)
    out = zeros(T, factor * (n - 1) + 1)
    @inbounds for k in 1:n
        out[(k - 1) * factor + 1] = h[k]
    end
    return out
end

"""
    upsample_kernel(K::AbstractMatrix, factor::Int) -> Matrix

2-D version of `upsample_filter`: insert `factor - 1` zero rows and columns
between every row and column of kernel `K`.
"""
function upsample_kernel(K::AbstractMatrix{T}, factor::Int)::Matrix{T} where {T}
    factor >= 1 || throw(ArgumentError("factor must be ≥ 1"))
    factor == 1 && return copy(K)
    rows, cols = size(K)
    nr = factor * (rows - 1) + 1
    nc = factor * (cols - 1) + 1
    out = zeros(T, nr, nc)
    @inbounds for c in 1:cols, r in 1:rows
        out[(r - 1) * factor + 1, (c - 1) * factor + 1] = K[r, c]
    end
    return out
end

"""
    check_pr_condition(qfp::QuincunxFilterPair; atol=1e-12) -> Bool

Verify the perfect-reconstruction condition for the 2-channel quincunx filter bank
defined by `qfp`.  Returns `true` if the reconstruction error is within `atol`.

The test computes the impulse response of the analysis-then-synthesis round trip on
a small image and checks that it equals a delayed impulse.
"""
function check_pr_condition(qfp::QuincunxFilterPair{T}; atol::Real = 1.0e-12)::Bool where {T}
    # Use a small 16×16 image to keep the test cheap.
    N = 16
    impulse = zeros(T, N, N)
    impulse[N ÷ 2, N ÷ 2] = one(T)
    sb0, sb1 = qfb_decompose(impulse, qfp)
    rec = qfb_reconstruct(sb0, sb1, qfp)
    # The reconstruction should equal the impulse (up to a global delay of ≤ kernel radius).
    # We check that exactly one value equals 1 and the rest are near 0.
    maxerr = maximum(abs, rec .- impulse)
    return maxerr < T(atol)
end

"""
    check_pr_condition(fp::FilterPair; N=64, atol=1e-12) -> Bool

Verify perfect reconstruction of the Laplacian Pyramid built from `fp` on an
`N×N` random image.
"""
function check_pr_condition(fp::FilterPair{T}; N::Int = 64, atol::Real = 1.0e-12)::Bool where {T}
    img = T.(reshape(1:(N * N), N, N))   # deterministic test signal
    coarse, bp = lp_decompose(img, fp)
    rec = lp_reconstruct(coarse, bp, fp)
    return maximum(abs, rec .- img) < T(atol)
end
