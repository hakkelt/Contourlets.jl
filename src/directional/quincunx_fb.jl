# Two-channel quincunx filter bank (QFB).
#
# After a shear operation the quincunx FB reduces to a 1-D two-channel
# filter bank in one spatial direction (Do & Vetterli 2005, §V):
#
#   dir=:col  →  filter along columns, keep even columns  (used after :h shear)
#   dir=:row  →  filter along rows,    keep even rows     (used after :v shear)
#                (implemented via transpose trick: pass image' to the :col path)
#
# Analysis (dir=:col):
#   lp[i,j] = Σ_l  h_q[l] · x[i, j − (l − c_h)]        (LP filter)
#   hp[i,j] = Σ_l  (−1)^(l−c_h) · h_q[l] · x[i, j−(l−c_h)]   (HP = modulated LP)
#   sb0[:,k] = lp[:,2k]   (keep even columns, 1-indexed as j=2,4,…)
#   sb1[:,k] = hp[:,2k]
#
# Synthesis (dir=:col):
#   for each k2, j_src = 2k2:
#     out[i, j_src + dc] += g_q[l] · sb0[i,k2]             (LP: dc = l − c_g)
#     out[i, j_src + dc] += (−1)^dc · g_q[l] · sb1[i,k2]  (HP)
#
# PR is exact (Haar pair H₀(z)=(1+z⁻¹)/2, G₀(z)=1+z satisfies the
# Nyquist(2) condition; see q2345.jl for derivation).

"""
    qfb_decompose(image, qfp; dir=:col) -> (sb0, sb1)

Two-channel quincunx filter bank analysis.

* `dir=:col` (default): filter in column direction, keep even columns.
* `dir=:row`: filter in row direction, keep even rows (via transpose).

Returns `(sb0, sb1)` where `sb0` is the LP subband and `sb1` the HP subband.

# Examples
```jldoctest
julia> using Contourlets, Random; Random.seed!(7)
julia> x = randn(16, 16)
julia> sb0, sb1 = qfb_decompose(x, Q2345)
julia> sb0_r, sb1_r = qfb_decompose(x, Q2345; dir=:row)
julia> size(sb0), size(sb0_r)
((16, 8), (8, 16))
```
"""
function qfb_decompose(image::AbstractMatrix, qfp::QuincunxFilterPair; dir::Symbol = :col)
    if dir === :col
        return _qfb_col_decompose(image, qfp)
    else
        sb0_t, sb1_t = _qfb_col_decompose(image', qfp)
        return collect(sb0_t'), collect(sb1_t')
    end
end

"""
    qfb_decompose!(sb0, sb1, image, qfp; dir=:col) -> (sb0, sb1)

In-place two-channel quincunx filter bank analysis.
"""
function qfb_decompose!(
        sb0::AbstractMatrix, sb1::AbstractMatrix,
        image::AbstractMatrix, qfp::QuincunxFilterPair;
        dir::Symbol = :col
    )
    if dir === :col
        return _qfb_col_decompose!(sb0, sb1, image, qfp)
    else
        sb0_t = collect(sb0')
        sb1_t = collect(sb1')
        _qfb_col_decompose!(sb0_t, sb1_t, image', qfp)
        sb0 .= sb0_t'
        sb1 .= sb1_t'
        return sb0, sb1
    end
end

"""
    qfb_reconstruct(sb0, sb1, qfp; dir=:col) -> image

Two-channel quincunx filter bank synthesis.

# Examples
```jldoctest
julia> using Contourlets, Random; Random.seed!(7)
julia> x = randn(16, 16)
julia> sb0, sb1 = qfb_decompose(x, Q2345)
julia> rec = qfb_reconstruct(sb0, sb1, Q2345)
julia> maximum(abs, rec .- x) < 1e-12
true
```
"""
function qfb_reconstruct(
        sb0::AbstractMatrix, sb1::AbstractMatrix,
        qfp::QuincunxFilterPair; dir::Symbol = :col
    )
    if dir === :col
        return _qfb_col_reconstruct(sb0, sb1, qfp)
    else
        rec_t = _qfb_col_reconstruct(collect(sb0'), collect(sb1'), qfp)
        return collect(rec_t')
    end
end

"""
    qfb_reconstruct!(out, sb0, sb1, qfp; dir=:col) -> out

In-place two-channel quincunx filter bank synthesis.
"""
function qfb_reconstruct!(
        out::AbstractMatrix, sb0::AbstractMatrix,
        sb1::AbstractMatrix, qfp::QuincunxFilterPair;
        dir::Symbol = :col
    )
    if dir === :col
        return _qfb_col_reconstruct!(out, sb0, sb1, qfp)
    else
        rec_t = _qfb_col_reconstruct(collect(sb0'), collect(sb1'), qfp)
        out .= rec_t'
        return out
    end
end

# ── Internal column-direction implementation ──────────────────────────────────

function _qfb_col_decompose(image::AbstractMatrix, qfp::QuincunxFilterPair)
    T = promote_type(eltype(image), eltype(qfp))
    img = T.(image)
    h = T.(vec(qfp.h_q))   # 1-D analysis LP filter
    c_h = qfp.c_h[2]          # column origin index (1-based)
    n1, n2 = size(img)
    d2 = n2 ÷ 2
    sb0 = zeros(T, n1, d2)
    sb1 = zeros(T, n1, d2)
    _qfb_col_decompose_kernel!(sb0, sb1, img, h, c_h, n1, n2, d2)
    return sb0, sb1
end

function _qfb_col_decompose!(
        sb0::AbstractMatrix, sb1::AbstractMatrix,
        image::AbstractMatrix, qfp::QuincunxFilterPair
    )
    h = eltype(sb0).(vec(qfp.h_q))
    c_h = qfp.c_h[2]
    n1, n2 = size(image)
    d2 = size(sb0, 2)
    _qfb_col_decompose_kernel!(sb0, sb1, image, h, c_h, n1, n2, d2)
    return sb0, sb1
end

@inline function _qfb_col_decompose_kernel!(sb0, sb1, img, h, c_h, n1, n2, d2)
    K = length(h)
    bv = Val(:symmetric)
    return @inbounds for k2 in 1:d2
        j = 2k2   # keep even columns (j=2,4,6,…)
        for i in 1:n1
            lp_val = zero(eltype(h))
            hp_val = zero(eltype(h))
            for l in 1:K
                dc = l - c_h          # column offset (may be negative)
                jj = j - dc           # column to read from image
                # symmetric boundary extension
                jj = _sym_idx(jj, n2)
                xval = img[i, jj]
                lp_val += h[l] * xval
                # HP = LP modulated by (-1)^dc
                hp_val += (iseven(dc) ? h[l] : -h[l]) * xval
            end
            sb0[i, k2] = lp_val
            sb1[i, k2] = hp_val
        end
    end
end

function _qfb_col_reconstruct(
        sb0::AbstractMatrix, sb1::AbstractMatrix,
        qfp::QuincunxFilterPair
    )
    T = promote_type(eltype(sb0), eltype(sb1), eltype(qfp))
    g = T.(vec(qfp.g_q))
    c_g = qfp.c_g[2]
    n1 = size(sb0, 1)
    n2 = size(sb0, 2) * 2
    out = zeros(T, n1, n2)
    _qfb_col_reconstruct_kernel!(out, T.(sb0), T.(sb1), g, c_g, n1, n2, size(sb0, 2))
    return out
end

function _qfb_col_reconstruct!(
        out::AbstractMatrix, sb0::AbstractMatrix,
        sb1::AbstractMatrix, qfp::QuincunxFilterPair
    )
    g = eltype(out).(vec(qfp.g_q))
    c_g = qfp.c_g[2]
    n1, n2 = size(out)
    d2 = size(sb0, 2)
    fill!(out, zero(eltype(out)))
    _qfb_col_reconstruct_kernel!(out, sb0, sb1, g, c_g, n1, n2, d2)
    return out
end

@inline function _qfb_col_reconstruct_kernel!(out, sb0, sb1, g, c_g, n1, n2, d2)
    K = length(g)
    return @inbounds for k2 in 1:d2
        j_src = 2k2   # source (even) column in output image
        for i in 1:n1
            s0 = sb0[i, k2]
            s1 = sb1[i, k2]
            for l in 1:K
                dc = l - c_g          # column offset
                j_out = j_src + dc       # scatter to output column
                if 1 <= j_out <= n2
                    # LP contribution
                    out[i, j_out] += g[l] * s0
                    # HP contribution: g_high = (-1)^dc * g
                    out[i, j_out] += (iseven(dc) ? g[l] : -g[l]) * s1
                end
            end
        end
    end
end

# Symmetric (reflect) boundary: _sym_idx(j, n) for j ∈ any integer, n ≥ 1.
@inline function _sym_idx(j::Int, n::Int)
    j < 1 && (j = 2 - j)
    j > n && (j = 2n - j)
    # Handle further out-of-bounds with a simple clamp
    j < 1 && (j = 1)
    j > n && (j = n)
    return j
end
