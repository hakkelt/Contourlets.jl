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
    Td = _data_eltype(bandpass)      # data type (real or complex)
    Tf = _filter_eltype(Td)          # real filter precision
    img = Td === eltype(bandpass) ? bandpass : Td.(bandpass)
    qup = _upsample_qfp_1d(qfp, 2^(tree_level - 1), Tf)
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
    Td = _data_eltype(subbands[1])   # data type (real or complex)
    Tf = _filter_eltype(Td)          # real filter precision
    qup = _upsample_qfp_1d(qfp, 2^(tree_level - 1), Tf)
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

# ── Allocation-free NSDFB (writes into pre-allocated `out`; `arena` supplies the
# internal-node scratch).  Used by the workspace `!` transform paths. ─────────

# Analysis: decompose `bandpass` into its `2^l_levels` subbands, written into the
# pre-allocated `out[1:2^l_levels]` (all the same size as `bandpass`).  `qup` is the
# precomputed upsampled filter bundle (from the workspace cache).
function _nsdfb_decompose_into!(
        out::Vector{<:AbstractMatrix}, bandpass::AbstractMatrix,
        l_levels::Int, qup, arena
    )
    l_levels == 0 && return (copyto!(out[1], bandpass); out)
    _nsdfb_split_into!(out, 1, bandpass, l_levels, 1, qup, arena)
    return out
end

function _nsdfb_split_into!(out, oi::Int, img, remaining::Int, depth::Int, qup, arena)
    dir = _nsdfb_direction(depth)
    if remaining == 1
        _nsqfb_decompose!(out[oi], out[oi + 1], img, qup, dir)
        return out
    end
    n1, n2 = size(img)
    sb0 = _arena_take!(arena, img, n1, n2)
    sb1 = _arena_take!(arena, img, n1, n2)
    _nsqfb_decompose!(sb0, sb1, img, qup, dir)
    half = 2^(remaining - 1)
    _nsdfb_split_into!(out, oi, sb0, remaining - 1, depth + 1, qup, arena)
    _nsdfb_split_into!(out, oi + half, sb1, remaining - 1, depth + 1, qup, arena)
    return out
end

# Synthesis: reconstruct the bandpass from `subbands` into the pre-allocated `dest`.
# `qup` is the precomputed upsampled filter bundle (from the workspace cache).
function _nsdfb_reconstruct_into!(
        dest::AbstractMatrix, subbands::Vector{<:AbstractMatrix},
        qup, arena
    )
    n = length(subbands)
    n == 1 && return (copyto!(dest, subbands[1]); dest)
    _nsdfb_merge_into!(dest, subbands, 1, n, round(Int, log2(n)), 1, qup, arena)
    return dest
end

function _nsdfb_merge_into!(dest, sbs, lo::Int, hi::Int, l::Int, depth::Int, qup, arena)
    dir = _nsdfb_direction(depth)
    if l == 1
        _nsqfb_reconstruct!(dest, sbs[lo], sbs[lo + 1], qup, dir)
        return dest
    end
    n1, n2 = size(dest)
    sb0 = _arena_take!(arena, dest, n1, n2)
    sb1 = _arena_take!(arena, dest, n1, n2)
    half = (hi - lo + 1) ÷ 2
    _nsdfb_merge_into!(sb0, sbs, lo, lo + half - 1, l - 1, depth + 1, qup, arena)
    _nsdfb_merge_into!(sb1, sbs, lo + half, hi, l - 1, depth + 1, qup, arena)
    _nsqfb_reconstruct!(dest, sb0, sb1, qup, dir)
    return dest
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

_nsqfb_decompose(image::AbstractMatrix, qup, dir::Tuple{Int, Int}) =
    _nsqfb_decompose!(similar(image), similar(image), image, qup, dir)

function _nsqfb_decompose!(
        sb0::AbstractMatrix, sb1::AbstractMatrix, image::AbstractMatrix,
        qup, dir::Tuple{Int, Int}
    )
    di, dj = dir
    n1, n2 = size(image)
    T = eltype(sb0)
    offs = qup.taps_h.offs
    v0 = qup.taps_h.vals0
    v1 = qup.taps_h.vals1
    dmin, dmax = qup.taps_h.dmin, qup.taps_h.dmax

    dimin = min(di * dmin, di * dmax)
    dimax = max(di * dmin, di * dmax)
    djmin = min(dj * dmin, dj * dmax)
    djmax = max(dj * dmin, dj * dmax)

    ilo = max(1, 1 + dimax)
    ihi = min(n1, n1 + dimin)
    jlo = max(1, 1 + djmax)
    jhi = min(n2, n2 + djmin)

    _nsqfb_decompose_kernel!(sb0, sb1, image, offs, v0, v1, di, dj, ilo, ihi, jlo, jhi)

    @inbounds for j in 1:n2, i in 1:n1
        if ilo <= i <= ihi && jlo <= j <= jhi
            continue
        end
        acc0 = zero(T)
        acc1 = zero(T)
        for t in eachindex(offs)
            d = offs[t]
            x = image[mod1(i - di * d, n1), mod1(j - dj * d, n2)]
            acc0 += v0[t] * x
            acc1 += v1[t] * x
        end
        sb0[i, j] = acc0
        sb1[i, j] = acc1
    end
    return sb0, sb1
end

function _nsqfb_decompose_kernel!(sb0::AbstractMatrix{T}, sb1::AbstractMatrix{T}, image::AbstractMatrix{T}, offs, v0, v1, di, dj, ilo, ihi, jlo, jhi) where {T <: Real}
    @turbo for j in jlo:jhi, i in ilo:ihi
        acc0 = zero(T)
        acc1 = zero(T)
        for t in eachindex(offs)
            d = offs[t]
            x = image[i - di * d, j - dj * d]
            acc0 += v0[t] * x
            acc1 += v1[t] * x
        end
        sb0[i, j] = acc0
        sb1[i, j] = acc1
    end
    return nothing
end

function _nsqfb_decompose_kernel!(sb0::AbstractMatrix{T}, sb1::AbstractMatrix{T}, image::AbstractMatrix{T}, offs, v0, v1, di, dj, ilo, ihi, jlo, jhi) where {T}
    @inbounds for j in jlo:jhi, i in ilo:ihi
        acc0 = zero(T)
        acc1 = zero(T)
        for t in eachindex(offs)
            d = offs[t]
            x = image[i - di * d, j - dj * d]
            acc0 += v0[t] * x
            acc1 += v1[t] * x
        end
        sb0[i, j] = acc0
        sb1[i, j] = acc1
    end
    return nothing
end

_nsqfb_reconstruct(sb0::AbstractMatrix, sb1::AbstractMatrix, qup, dir::Tuple{Int, Int}) =
    _nsqfb_reconstruct!(similar(sb0), sb0, sb1, qup, dir)

function _nsqfb_reconstruct!(
        out::AbstractMatrix, sb0::AbstractMatrix, sb1::AbstractMatrix, qup,
        dir::Tuple{Int, Int}
    )
    di, dj = dir
    n1, n2 = size(sb0)
    T = eltype(out)
    offs = qup.taps_g.offs
    v0 = qup.taps_g.vals0
    v1 = qup.taps_g.vals1
    dmin, dmax = qup.taps_g.dmin, qup.taps_g.dmax
    scale = qup.scale

    dimin = min(di * dmin, di * dmax)
    dimax = max(di * dmin, di * dmax)
    djmin = min(dj * dmin, dj * dmax)
    djmax = max(dj * dmin, dj * dmax)

    ilo = max(1, 1 + dimax)
    ihi = min(n1, n1 + dimin)
    jlo = max(1, 1 + djmax)
    jhi = min(n2, n2 + djmin)

    _nsqfb_reconstruct_kernel!(out, sb0, sb1, offs, v0, v1, scale, di, dj, ilo, ihi, jlo, jhi)

    @inbounds for j in 1:n2, i in 1:n1
        if ilo <= i <= ihi && jlo <= j <= jhi
            continue
        end
        acc = zero(T)
        for t in eachindex(offs)
            d = offs[t]
            x0 = sb0[mod1(i - di * d, n1), mod1(j - dj * d, n2)]
            x1 = sb1[mod1(i - di * d, n1), mod1(j - dj * d, n2)]
            acc += v0[t] * x0 + v1[t] * x1
        end
        out[i, j] = scale * acc
    end
    return out
end

function _nsqfb_reconstruct_kernel!(out::AbstractMatrix{T}, sb0::AbstractMatrix{T}, sb1::AbstractMatrix{T}, offs, v0, v1, scale, di, dj, ilo, ihi, jlo, jhi) where {T <: Real}
    @turbo for j in jlo:jhi, i in ilo:ihi
        acc = zero(T)
        for t in eachindex(offs)
            d = offs[t]
            x0 = sb0[i - di * d, j - dj * d]
            x1 = sb1[i - di * d, j - dj * d]
            acc += v0[t] * x0 + v1[t] * x1
        end
        out[i, j] = scale * acc
    end
    return nothing
end

function _nsqfb_reconstruct_kernel!(out::AbstractMatrix{T}, sb0::AbstractMatrix{T}, sb1::AbstractMatrix{T}, offs, v0, v1, scale, di, dj, ilo, ihi, jlo, jhi) where {T}
    @inbounds for j in jlo:jhi, i in ilo:ihi
        acc = zero(T)
        for t in eachindex(offs)
            d = offs[t]
            x0 = sb0[i - di * d, j - dj * d]
            x1 = sb1[i - di * d, j - dj * d]
            acc += v0[t] * x0 + v1[t] * x1
        end
        out[i, j] = scale * acc
    end
    return nothing
end

# ── Helpers ───────────────────────────────────────────────────────────────────

function _build_compact_taps(f0::Vector{T}, c0::Int, f1::Vector{T}, c1::Int, factor::Int) where {T}
    dmin_lag = min(1 - c0, 1 - c1)
    dmax_lag = max(length(f0) - c0, length(f1) - c1)

    offs = Int[]
    v0 = T[]
    v1 = T[]
    for d in dmin_lag:dmax_lag
        m0 = d + c0
        m1 = d + c1
        val0 = (1 <= m0 <= length(f0)) ? f0[m0] : zero(T)
        val1 = (1 <= m1 <= length(f1)) ? f1[m1] : zero(T)
        if val0 != zero(T) || val1 != zero(T)
            push!(offs, d * factor)
            push!(v0, val0)
            push!(v1, val1)
        end
    end
    dmin = isempty(offs) ? 0 : minimum(offs)
    dmax = isempty(offs) ? 0 : maximum(offs)
    return _CompactTapsPair{T}(v0, v1, offs, dmin, dmax)
end

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

    taps_h = _build_compact_taps(h0, c0, h1, c1, factor)
    taps_g = _build_compact_taps(g0, cg0, g1, cg1, factor)

    if factor > 1
        h0 = upsample_filter(h0, factor); c0 = (c0 - 1) * factor + 1
        h1 = upsample_filter(h1, factor); c1 = (c1 - 1) * factor + 1
        g0 = upsample_filter(g0, factor); cg0 = (cg0 - 1) * factor + 1
        g1 = upsample_filter(g1, factor); cg1 = (cg1 - 1) * factor + 1
    end
    return (
        h0 = h0, c0 = c0, h1 = h1, c1 = c1,
        g0 = g0, cg0 = cg0, g1 = g1, cg1 = cg1, scale = scale,
        taps_h = taps_h, taps_g = taps_g,
    )
end

# ── FFT-Domain Equivalent Filters (D2) ───────────────────────────────────────

function _build_filter_grid(T, n1, n2, offs, vals, di, dj)
    grid = zeros(T, n1, n2)
    for t in eachindex(offs)
        d = offs[t]
        grid[mod1(1 + di * d, n1), mod1(1 + dj * d, n2)] += vals[t]
    end
    return grid
end

function _get_transfer_functions(T, n1, n2, qup, depth; is_real = true)
    di, dj = _nsdfb_direction(depth)
    h0 = _build_filter_grid(T, n1, n2, qup.taps_h.offs, qup.taps_h.vals0, di, dj)
    h1 = _build_filter_grid(T, n1, n2, qup.taps_h.offs, qup.taps_h.vals1, di, dj)
    g0 = _build_filter_grid(T, n1, n2, qup.taps_g.offs, qup.taps_g.vals0, di, dj)
    g1 = _build_filter_grid(T, n1, n2, qup.taps_g.offs, qup.taps_g.vals1, di, dj)
    if is_real
        return rfft(h0), rfft(h1), rfft(g0), rfft(g1)
    else
        return fft(h0), fft(h1), fft(g0), fft(g1)
    end
end

function _build_equivalent_filters(T, n1, n2, qup, L, is_real)
    H_leaves = Matrix{Complex{T}}[]
    G_leaves = Matrix{Complex{T}}[]

    H0s, H1s, G0s, G1s = Matrix{Complex{T}}[], Matrix{Complex{T}}[], Matrix{Complex{T}}[], Matrix{Complex{T}}[]
    for d in 1:L
        h0, h1, g0, g1 = _get_transfer_functions(T, n1, n2, qup, d; is_real = is_real)
        push!(H0s, h0); push!(H1s, h1)
        push!(G0s, g0); push!(G1s, g1)
    end

    function build_branch(depth, current_H, current_G)
        if depth > L
            push!(H_leaves, current_H)
            push!(G_leaves, current_G)
            return
        end
        build_branch(depth + 1, current_H .* H0s[depth], current_G .* G0s[depth])
        build_branch(depth + 1, current_H .* H1s[depth], current_G .* G1s[depth])
        return nothing
    end

    init_shape = is_real ? (n1 ÷ 2 + 1, n2) : (n1, n2)
    build_branch(1, ones(Complex{T}, init_shape), ones(Complex{T}, init_shape) .* (qup.scale^L))

    return H_leaves, G_leaves
end
