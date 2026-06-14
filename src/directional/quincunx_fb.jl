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
julia> using Contourlets, Random

julia> x = randn(Xoshiro(7), 16, 16);

julia> sb0, sb1 = qfb_decompose(x, Q2345);

julia> sb0_r, sb1_r = qfb_decompose(x, Q2345; dir = :row);

julia> size(sb0), size(sb0_r)
((16, 8), (8, 16))
```
"""
function qfb_decompose(image::AbstractMatrix, qfp::QuincunxFilterPair; dir::Symbol = :col)
    T = promote_type(eltype(image), eltype(qfp))
    img = T === eltype(image) ? image : T.(image)
    qfpT = _convert_qfp(T, qfp)
    if dir === :col
        return _qfb_col_decompose(img, qfpT)
    else
        sb0_t, sb1_t = _qfb_col_decompose(img', qfpT)
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
julia> using Contourlets, Random

julia> x = randn(Xoshiro(7), 16, 16);

julia> sb0, sb1 = qfb_decompose(x, Q2345);

julia> rec = qfb_reconstruct(sb0, sb1, Q2345);

julia> maximum(abs, rec .- x) < 1e-12
true
```
"""
function qfb_reconstruct(
        sb0::AbstractMatrix, sb1::AbstractMatrix,
        qfp::QuincunxFilterPair; dir::Symbol = :col
    )
    T = promote_type(eltype(sb0), eltype(sb1), eltype(qfp))
    sb0T = T === eltype(sb0) ? sb0 : T.(sb0)
    sb1T = T === eltype(sb1) ? sb1 : T.(sb1)
    qfpT = _convert_qfp(T, qfp)
    if dir === :col
        return _qfb_col_reconstruct(sb0T, sb1T, qfpT)
    else
        rec_t = _qfb_col_reconstruct(collect(sb0T'), collect(sb1T'), qfpT)
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

# Internal: callers (the public `qfb_decompose`) pass an `img`/`qfp` already
# promoted to a common element type `T`.
function _qfb_col_decompose(img::AbstractMatrix{T}, qfp::QuincunxFilterPair{T}) where {T}
    n1, n2 = size(img)
    d2 = n2 ÷ 2
    sb0 = zeros(T, n1, d2)
    sb1 = zeros(T, n1, d2)
    return _qfb_col_decompose!(sb0, sb1, img, qfp)
end

function _qfb_col_decompose!(
        sb0::AbstractMatrix, sb1::AbstractMatrix,
        image::AbstractMatrix, qfp::QuincunxFilterPair
    )
    n1, n2 = size(image)
    d2 = size(sb0, 2)
    if is_ladder(qfp)
        iseven(n2) || throw(ArgumentError("ladder QFB requires an even number of columns"))
        _qfb_col_decompose_ladder!(sb0, sb1, image, qfp)
    else
        h = eltype(sb0).(vec(qfp.h_q))
        c_h = qfp.c_h[2]
        _qfb_col_decompose_kernel!(sb0, sb1, image, h, c_h, n1, n2, d2)
    end
    return sb0, sb1
end

@inline function _qfb_col_decompose_kernel!(sb0, sb1, img, h, c_h, n1, n2, d2)
    K = length(h)
    return @inbounds for k2 in 1:d2
        j = 2k2   # keep even columns (j=2,4,6,…)
        for i in 1:n1
            lp_val = zero(eltype(h))
            hp_val = zero(eltype(h))
            for l in 1:K
                dc = l - c_h          # column offset (may be negative)
                jj = j - dc           # column to read from image
                # symmetric boundary extension (shared helper from conv2d.jl)
                jj = _clamp_idx(jj, n2, Val(:symmetric))
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

# Internal: callers (the public `qfb_reconstruct`) pass `sb0`/`sb1`/`qfp` already
# promoted to a common element type `T`.
function _qfb_col_reconstruct(
        sb0::AbstractMatrix{T}, sb1::AbstractMatrix{T},
        qfp::QuincunxFilterPair{T}
    ) where {T}
    n1 = size(sb0, 1)
    n2 = size(sb0, 2) * 2
    out = zeros(T, n1, n2)
    return _qfb_col_reconstruct!(out, sb0, sb1, qfp)
end

function _qfb_col_reconstruct!(
        out::AbstractMatrix, sb0::AbstractMatrix,
        sb1::AbstractMatrix, qfp::QuincunxFilterPair
    )
    n1, n2 = size(out)
    d2 = size(sb0, 2)
    fill!(out, zero(eltype(out)))
    if is_ladder(qfp)
        _qfb_col_reconstruct_ladder!(out, sb0, sb1, qfp)
    else
        g = eltype(out).(vec(qfp.g_q))
        c_g = qfp.c_g[2]
        _qfb_col_reconstruct_kernel!(out, sb0, sb1, g, c_g, n1, n2, d2)
    end
    return out
end

# ── Ladder (lifting) realisation — structural PR, periodic extension ─────────
#
# Channel 0 lives on the odd image columns (1-based), channel 1 on the even
# columns.  See src/filters/q2345.jl for the ladder derivation.

function _qfb_col_decompose_ladder!(
        sb0::AbstractMatrix, sb1::AbstractMatrix,
        img::AbstractMatrix, qfp::QuincunxFilterPair
    )
    T = eltype(sb0)
    f = _ladder_modulate(T.(qfp.f_ladder))
    c = length(f) ÷ 2
    n1 = size(img, 1)
    d2 = size(sb0, 2)
    s2 = sqrt(T(2))
    # y0 = (1/√2)·(p0 − B₁(p1)),  B₁ lags m − c + 1
    @inbounds for k in 1:d2
        for i in 1:n1
            acc = zero(T)
            for m in eachindex(f)
                kk = mod1(k - (m - c + 1), d2)
                acc += f[m] * img[i, 2kk]
            end
            sb0[i, k] = (img[i, 2k - 1] - acc) / s2
        end
    end
    # y1 = −√2·p1 − B₀(y0),  B₀ lags m − c
    @inbounds for k in 1:d2
        for i in 1:n1
            acc = zero(T)
            for m in eachindex(f)
                kk = mod1(k - (m - c), d2)
                acc += f[m] * sb0[i, kk]
            end
            sb1[i, k] = -s2 * img[i, 2k] - acc
        end
    end
    return sb0, sb1
end

function _qfb_col_reconstruct_ladder!(
        out::AbstractMatrix, sb0::AbstractMatrix,
        sb1::AbstractMatrix, qfp::QuincunxFilterPair
    )
    T = eltype(out)
    f = _ladder_modulate(T.(qfp.f_ladder))
    c = length(f) ÷ 2
    n1 = size(out, 1)
    d2 = size(sb0, 2)
    s2 = sqrt(T(2))
    # p1 = (−1/√2)·(y1 + B₀(y0)) → even columns
    @inbounds for k in 1:d2
        for i in 1:n1
            acc = zero(T)
            for m in eachindex(f)
                kk = mod1(k - (m - c), d2)
                acc += f[m] * sb0[i, kk]
            end
            out[i, 2k] = -(sb1[i, k] + acc) / s2
        end
    end
    # p0 = √2·y0 + B₁(p1) → odd columns
    @inbounds for k in 1:d2
        for i in 1:n1
            acc = zero(T)
            for m in eachindex(f)
                kk = mod1(k - (m - c + 1), d2)
                acc += f[m] * out[i, 2kk]
            end
            out[i, 2k - 1] = s2 * sb0[i, k] + acc
        end
    end
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
