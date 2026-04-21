# Nonsubsampled Directional Filter Bank (NSDFB).
#
# The NSDFB removes all downsampling from the DFB binary tree.  At pyramid
# level `tree_level` (1 = finest), the 1-D column analysis filters are
# upsampled by factor F = 2^(tree_level-1) via zero-insertion (à trous).
#
# PR for the non-decimated 2-channel FB (da Cunha et al. 2006):
#   Synthesis filters:  G_k(z) = H_k(z^{-1})  (time-reversed analysis)
#   Reconstruction:     out = Σ_k (G_k * sb_k)
#
# For symmetric FIR filters H_k(z^{-1}) = H_k(z), so the synthesis uses the
# same tap values as analysis but applied with time-reversed indices (+dc).
#
# The HP filter is the base-level modulation H_1(z) = H_0(-z), THEN upsampled
# (not modulation of the already-upsampled LP), because modulating an
# upsampled filter that has zeros at odd dc positions yields H_1 = H_0.
#
# PR is verified analytically for the Haar pair at all tree_levels ≥ 1.

"""
    nsdfb_decompose(bandpass, l_levels, qfp, tree_level) -> Vector{Matrix}

Nonsubsampled DFB analysis.  Returns `2^l_levels` directional subbands,
each the same size as `bandpass`.  `tree_level` (≥ 1) controls the
à trous upsampling factor `2^(tree_level-1)` applied to the 1-D filters.

# Examples
```jldoctest
julia> using Contourlets, Random; Random.seed!(5)
julia> x = randn(16, 16)
julia> sbs = nsdfb_decompose(x, 2, Q2345, 1)
julia> length(sbs)
4
julia> size(sbs[1]) == size(x)
true
```
"""
function nsdfb_decompose(
        bandpass::AbstractMatrix, l_levels::Int,
        qfp::QuincunxFilterPair, tree_level::Int
    )
    l_levels >= 0 || throw(ArgumentError("l_levels must be ≥ 0"))
    tree_level >= 1 || throw(ArgumentError("tree_level must be ≥ 1"))
    l_levels == 0 && return [copy(bandpass)]
    T = promote_type(eltype(bandpass), eltype(qfp))
    img = T.(bandpass)
    qup = _upsample_qfp_1d(qfp, 2^(tree_level - 1), T)
    return _nsdfb_split(img, l_levels, 1, qup)
end

function _nsdfb_split(img::AbstractMatrix, remaining::Int, depth::Int, qup)
    shear_dir = isodd(depth) ? :h : :v
    sh = shear(img, shear_dir)
    sb0, sb1 = _nsqfb_decompose(sh, qup)
    sb0 = inv_shear(sb0, shear_dir)   # allocating — avoids src==dst aliasing
    sb1 = inv_shear(sb1, shear_dir)
    remaining == 1 && return [sb0, sb1]
    return vcat(
        _nsdfb_split(sb0, remaining - 1, depth + 1, qup),
        _nsdfb_split(sb1, remaining - 1, depth + 1, qup)
    )
end

"""
    nsdfb_reconstruct(subbands, qfp, tree_level) -> bandpass

Nonsubsampled DFB synthesis.

# Examples
```jldoctest
julia> using Contourlets, Random; Random.seed!(5)
julia> x = randn(16, 16)
julia> sbs = nsdfb_decompose(x, 2, Q2345, 1)
julia> rec = nsdfb_reconstruct(sbs, Q2345, 1)
julia> maximum(abs, rec .- x) < 1e-12
true
```
"""
function nsdfb_reconstruct(
        subbands::Vector{<:AbstractMatrix},
        qfp::QuincunxFilterPair, tree_level::Int
    )
    n = length(subbands)
    ispow2(n) || throw(ArgumentError("number of subbands must be a power of 2"))
    n == 1 && return copy(subbands[1])
    T = promote_type(eltype(subbands[1]), eltype(qfp))
    qup = _upsample_qfp_1d(qfp, 2^(tree_level - 1), T)
    l_levels = round(Int, log2(n))
    return _nsdfb_merge(subbands, l_levels, 1, qup)
end

function _nsdfb_merge(sbs::Vector{<:AbstractMatrix}, l::Int, depth::Int, qup)
    shear_dir = isodd(depth) ? :h : :v
    half = length(sbs) ÷ 2
    if l == 1
        sb0, sb1 = sbs[1], sbs[2]
    else
        sb0 = _nsdfb_merge(sbs[1:half], l - 1, depth + 1, qup)
        sb1 = _nsdfb_merge(sbs[(half + 1):end], l - 1, depth + 1, qup)
    end
    sh0 = shear(sb0, shear_dir)
    sh1 = shear(sb1, shear_dir)
    rec = _nsqfb_reconstruct(sh0, sh1, qup)
    return inv_shear(rec, shear_dir)   # allocating — avoids src==dst aliasing
end

# ── Non-decimated 1-D column filter bank ─────────────────────────────────────
#
# `qup` is a NamedTuple (h, h_high, c_h, K) where:
#   h      = à trous upsampled analysis LP filter
#   h_high = à trous upsampled analysis HP filter (= upsample(base HP))
#   c_h    = center index of h (and h_high), 1-based
#   K      = length of h (and h_high)
#
# Analysis (column direction, periodic extension):
#   sb0[i,j] = Σ_m  h[m]        · x[i, mod1(j − dm, n2)]   LP   dm = m − c_h
#   sb1[i,j] = Σ_m  h_high[m]   · x[i, mod1(j − dm, n2)]   HP
#
# Synthesis (time-reversed = correlation form, j+dm instead of j−dm):
#   out[i,j] = Σ_m  h[m]        · sb0[i, mod1(j + dm, n2)]   LP syn = H_0(z^{-1})
#            + Σ_m  h_high[m]   · sb1[i, mod1(j + dm, n2)]   HP syn = H_1(z^{-1})
#
# PR: H_0(z)·H_0(z^{-1}) + H_1(z)·H_1(z^{-1}) = |H_0|²+|H_1|² = 1
#   (holds for the Haar pair at all upsampling factors, verified analytically).

function _nsqfb_decompose(image::AbstractMatrix, qup)
    h, h_high, c_h, K = qup.h, qup.h_high, qup.c_h, qup.K
    n1, n2 = size(image)
    sb0 = similar(image); sb1 = similar(image)
    @inbounds for j in 1:n2, i in 1:n1
        lp = zero(eltype(image)); hp = zero(eltype(image))
        for m in 1:K
            dm = m - c_h
            jj = mod1(j - dm, n2)   # periodic boundary
            xval = image[i, jj]
            lp += h[m] * xval
            hp += h_high[m] * xval
        end
        sb0[i, j] = lp
        sb1[i, j] = hp
    end
    return sb0, sb1
end

function _nsqfb_reconstruct(sb0::AbstractMatrix, sb1::AbstractMatrix, qup)
    h, h_high, c_h, K = qup.h, qup.h_high, qup.c_h, qup.K
    n1, n2 = size(sb0)
    out = zeros(eltype(sb0), n1, n2)
    @inbounds for j in 1:n2, i in 1:n1
        lp = zero(eltype(sb0)); hp = zero(eltype(sb0))
        for m in 1:K
            dm = m - c_h
            jj = mod1(j + dm, n2)   # time-reversed = +dm
            lp += h[m] * sb0[i, jj]
            hp += h_high[m] * sb1[i, jj]
        end
        out[i, j] = lp + hp
    end
    return out
end

# ── Helpers ───────────────────────────────────────────────────────────────────

"""
Build the `qup` NamedTuple for the NSDFB at a given `factor`:
  h      = upsample(h_q_base, factor)
  h_high = upsample(h_high_base, factor)     (base HP: h_q modulated before upsampling)
  c_h    = upsampled center index
  K      = filter length after upsampling
"""
function _upsample_qfp_1d(qfp::QuincunxFilterPair, factor::Int, ::Type{T}) where {T}
    h_base = T.(vec(qfp.h_q))
    c_base = qfp.c_h[2]
    K_base = length(h_base)
    # Build base HP BEFORE upsampling (column modulation by (-1)^{dc})
    h_high_base = T[iseven(l - c_base) ? h_base[l] : -h_base[l] for l in 1:K_base]

    if factor == 1
        return (h = h_base, h_high = h_high_base, c_h = c_base, K = K_base)
    end
    h_up = upsample_filter(h_base, factor)
    h_high_up = upsample_filter(h_high_base, factor)
    c_up = (c_base - 1) * factor + 1
    return (h = h_up, h_high = h_high_up, c_h = c_up, K = length(h_up))
end
