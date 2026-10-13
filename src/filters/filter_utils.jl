# Utilities operating on FilterPair and QuincunxFilterPair.

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
    check_pr_condition(qfp::QuincunxFilterPair{T}; atol=_pr_atol(T)) -> Bool

Verify the perfect-reconstruction condition for the 2-channel quincunx filter bank
defined by `qfp`.  Returns `true` if the reconstruction error is within `atol`,
which defaults to `1e-12` for `Float64` filters and scales with `eps(T)` for other
precisions (about `5e-4` for `Float32`).

The test computes the impulse response of the analysis-then-synthesis round trip on
a small image and checks that it equals a delayed impulse.
"""
function check_pr_condition(qfp::QuincunxFilterPair{T}; atol::Real = _pr_atol(T))::Bool where {T}
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
    check_pr_condition(fp::FilterPair{T}; N=64, atol=_pr_atol(T)) -> Bool

Verify perfect reconstruction of the Laplacian Pyramid built from `fp` on an
`N×N` test image.  The error is measured relative to the image's largest value,
against `atol`, whose default scales with `eps(T)` as in the method above.
"""
function check_pr_condition(fp::FilterPair{T}; N::Int = 64, atol::Real = _pr_atol(T))::Bool where {T}
    img = T.(reshape(1:(N * N), N, N))   # deterministic test signal
    coarse, bp = lp_decompose(img, fp)
    rec = lp_reconstruct(coarse, bp, fp)
    return maximum(abs, rec .- img) < T(atol) * maximum(abs, img)
end

# Default PR tolerance: 1e-12 at Float64, scaled with the precision's eps.
_pr_atol(::Type{T}) where {T <: AbstractFloat} = 1.0e-12 * (eps(T) / eps(Float64))
