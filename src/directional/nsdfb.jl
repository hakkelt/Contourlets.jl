# Nonsubsampled Directional Filter Bank (NSDFB).
#
# The NSDFB removes all downsampling from the DFB binary tree.  At pyramid
# level `tree_level` (1 = finest), the 1-D analysis filters are upsampled by
# factor F = 2^(tree_level-1) via zero-insertion (à trous).
#
# Each tree depth splits the input with a two-channel filter bank whose 1-D
# filters are applied along one of four lattice directions (periodic / circular
# convolution on the image torus), cycling with depth:
#
#   depth ≡ 1 (mod 4):  (0, 1)   across columns  (horizontal frequency split)
#   depth ≡ 2 (mod 4):  (1, 0)   across rows     (vertical frequency split)
#   depth ≡ 3 (mod 4):  (1, 1)   main diagonal
#   depth ≡ 0 (mod 4):  (1, −1)  anti-diagonal
#
# Because every stage is a circular convolution, the NSDFB — and hence the
# NSCT — is exactly invariant under circular shifts of the input.
#
# PR for the non-decimated 2-channel FB (da Cunha et al. 2006):
#   Reconstruction:  out = scale · Σ_k (G_k * sb_k)
#   with scale · (G₀H₀ + G₁H₁) = 1.
#
# Ladder mode (default Q2345): the equivalent analysis/synthesis filters of the
# Phoong lifting network satisfy G₀H₀ + G₁H₁ = 2 structurally, so scale = 1/2.
# Modulation mode: G_k(z) = H_k(z^{-1}) and scale = 1, which is PR whenever
# |H₀|² + |H₁|² = 1 (e.g. the Haar pair).  In both modes the identity is
# preserved by à trous upsampling and holds along any lattice direction, so PR
# holds at every depth and tree_level ≥ 1.

# Filtering direction (di, dj) for a given tree depth.
function _nsdfb_direction(depth::Int)
    r = mod1(depth, 4)
    r == 1 && return (0, 1)
    r == 2 && return (1, 0)
    r == 3 && return (1, 1)
    return (1, -1)
end

"""
    nsdfb_decompose(bandpass, l_levels, qfp, tree_level) -> Vector{Matrix}

Nonsubsampled DFB analysis.  Returns `2^l_levels` directional subbands,
each the same size as `bandpass`.  `tree_level` (≥ 1) controls the
à trous upsampling factor `2^(tree_level-1)` applied to the 1-D filters.

# Examples
```jldoctest
julia> using Contourlets, Random

julia> x = randn(Xoshiro(5), 16, 16);

julia> sbs = nsdfb_decompose(x, 2, Q2345, 1);

julia> length(sbs), size(sbs[1]) == size(x)
(4, true)
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
    sb0, sb1 = _nsqfb_decompose(img, qup, _nsdfb_direction(depth))
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
julia> using Contourlets, Random

julia> x = randn(Xoshiro(5), 16, 16);

julia> sbs = nsdfb_decompose(x, 2, Q2345, 1);

julia> rec = nsdfb_reconstruct(sbs, Q2345, 1);

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
    half = length(sbs) ÷ 2
    if l == 1
        sb0, sb1 = sbs[1], sbs[2]
    else
        sb0 = _nsdfb_merge(sbs[1:half], l - 1, depth + 1, qup)
        sb1 = _nsdfb_merge(sbs[(half + 1):end], l - 1, depth + 1, qup)
    end
    return _nsqfb_reconstruct(sb0, sb1, qup, _nsdfb_direction(depth))
end

# ── Non-decimated 1-D directional filter bank ────────────────────────────────
#
# `qup` is a NamedTuple (h0, c0, h1, c1, g0, cg0, g1, cg1, scale) holding the
# à trous upsampled equivalent analysis filters (h0, h1) and synthesis filters
# (g0, g1) with their 1-based origin indices, plus a reconstruction scale.
#
# Analysis along lattice direction (di, dj), periodic extension:
#   sb_k[i,j] = Σ_m  h_k[m] · x[mod1(i − di·(m − c_k), n1), mod1(j − dj·(m − c_k), n2)]
#
# Synthesis:
#   out[i,j] = scale · Σ_k Σ_m  g_k[m] · sb_k[mod1(i − di·(m − cg_k), n1), …]
#
# PR condition: scale · (G₀(z)H₀(z) + G₁(z)H₁(z)) = 1.
#   • Modulation mode: G_k = H_k(z^{-1}), scale = 1 (holds e.g. for Haar).
#   • Ladder mode: G_k from the inverse ladder, scale = 1/2 (structural;
#     follows from the decimated PR identity G₀H₀ + G₁H₁ = 2).
# Both identities are invariant under à trous upsampling (z → z^{2^l}).

function _nsqfb_decompose(image::AbstractMatrix, qup, dir::Tuple{Int, Int})
    h0, c0, h1, c1 = qup.h0, qup.c0, qup.h1, qup.c1
    di, dj = dir
    n1, n2 = size(image)
    sb0 = similar(image); sb1 = similar(image)
    @inbounds for j in 1:n2, i in 1:n1
        acc0 = zero(eltype(image))
        for m in eachindex(h0)
            d = m - c0
            acc0 += h0[m] * image[mod1(i - di * d, n1), mod1(j - dj * d, n2)]
        end
        acc1 = zero(eltype(image))
        for m in eachindex(h1)
            d = m - c1
            acc1 += h1[m] * image[mod1(i - di * d, n1), mod1(j - dj * d, n2)]
        end
        sb0[i, j] = acc0
        sb1[i, j] = acc1
    end
    return sb0, sb1
end

function _nsqfb_reconstruct(
        sb0::AbstractMatrix, sb1::AbstractMatrix, qup,
        dir::Tuple{Int, Int}
    )
    g0, cg0, g1, cg1 = qup.g0, qup.cg0, qup.g1, qup.cg1
    di, dj = dir
    n1, n2 = size(sb0)
    out = zeros(eltype(sb0), n1, n2)
    @inbounds for j in 1:n2, i in 1:n1
        acc = zero(eltype(sb0))
        for m in eachindex(g0)
            d = m - cg0
            acc += g0[m] * sb0[mod1(i - di * d, n1), mod1(j - dj * d, n2)]
        end
        for m in eachindex(g1)
            d = m - cg1
            acc += g1[m] * sb1[mod1(i - di * d, n1), mod1(j - dj * d, n2)]
        end
        out[i, j] = qup.scale * acc
    end
    return out
end

# ── Helpers ───────────────────────────────────────────────────────────────────

"""
Build the `qup` NamedTuple for the NSDFB at a given à trous `factor`.

Ladder mode: equivalent filters from the lifting network (see q2345.jl),
reconstruction scale 1/2.  Modulation mode: HP by modulation of `h_q`,
synthesis = time-reversed analysis, scale 1.
"""
function _upsample_qfp_1d(qfp::QuincunxFilterPair, factor::Int, ::Type{T}) where {T}
    if is_ladder(qfp)
        eq = _ladder_equivalent_filters(_ladder_modulate(T.(qfp.f_ladder)))
        h0 = eq.h0; c0 = eq.c0; h1 = eq.h1; c1 = eq.c1
        g0 = eq.g0; cg0 = eq.cg0; g1 = eq.g1; cg1 = eq.cg1
        scale = T(0.5)
    else
        h0 = T.(vec(qfp.h_q))
        c0 = qfp.c_h[2]
        K = length(h0)
        # HP by modulation BEFORE upsampling (modulating an à trous filter is a no-op)
        h1 = T[iseven(m - c0) ? h0[m] : -h0[m] for m in 1:K]
        c1 = c0
        # synthesis = time-reversed analysis: G_k(z) = H_k(z^{-1})
        g0 = reverse(h0); cg0 = K + 1 - c0
        g1 = reverse(h1); cg1 = K + 1 - c1
        scale = one(T)
    end
    if factor > 1
        h0 = upsample_filter(h0, factor); c0 = (c0 - 1) * factor + 1
        h1 = upsample_filter(h1, factor); c1 = (c1 - 1) * factor + 1
        g0 = upsample_filter(g0, factor); cg0 = (cg0 - 1) * factor + 1
        g1 = upsample_filter(g1, factor); cg1 = (cg1 - 1) * factor + 1
    end
    return (
        h0 = h0, c0 = c0, h1 = h1, c1 = c1,
        g0 = g0, cg0 = cg0, g1 = g1, cg1 = cg1, scale = scale,
    )
end
