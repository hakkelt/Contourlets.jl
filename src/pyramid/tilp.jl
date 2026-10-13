# Translation-invariant Laplacian pyramid (TILP) of Eslami & Radha (2006, §III-A).
#
# Do & Vetterli (2003) write one level of the Laplacian pyramid as an oversampled
# filter bank in the polyphase domain (Eslami & Radha, Fig. 5, eq. 1): with
# M = 2I and the coset representatives k_c ∈ {(0,0), (1,0), (0,1), (1,1)}, the
# coarse channel is H(z) and the four detail channels are the polyphase
# components of the detail image,
#
#     K_c(z) = z^(k_c) − G_c(z^M) H(z),        G_c(z) = ↓M{z^(k_c) G(z)},
#
# each followed by ↓M.  The synthesis is the frame reconstruction of Do &
# Vetterli, x = G·↑M c + Σ_c F_c·↑M y_c with
#
#     F_c(z) = z^(−k_c) − G(z) H_c(z^M),       H_c(z) = ↓M{z^(−k_c) H(z)}.
#
# The TILP removes every sampler.  By the generalized algorithme à trous
# (Eslami & Radha, Proposition 1 and Corollary 1) level i then filters with
# H(z^(M^(i−1))), K_c(z^(M^(i−1))), …, and each synthesis level carries a factor
# 1/|det M| = 1/4.  Perfect reconstruction needs only that h and g be
# biorthogonal (their Proposition 2): ¼(G H + Σ_c F_c K_c) = 1.  The redundancy is
# 4L + 1.  Downsampling the level-i channels by M^i gives back, for every shift
# of the sampling lattice, the decimated pyramid's channels, so the TILP is the
# decimated pyramid made translation invariant, aliasing included.
#
# Here every filter is evaluated in closed form on the DFT grid — the separable
# 2-D filters are products of 1-D DTFTs, and the polyphase components follow from
# G_c(Mω) = ¼ Σ_{a ∈ {0,π}²} e^{j(ω+a)·k_c} G(ω+a) — and the cascade of a level is
# folded into one mask per channel, so a TILP channel is one pointwise product on
# the spectrum.  Filtering on the DFT grid is circular convolution: every channel
# commutes with circular shifts of the input.
#
# References:
#   R. Eslami & H. Radha, "Translation-invariant contourlet transform and its
#     application to image denoising", IEEE TIP 15(11), 2006, §III-A.
#   M. N. Do & M. Vetterli, "Framing pyramids", IEEE TSP 51(9), 2003.

# Coset representatives k_c of M = 2I, in (row, column) order.
const _TILP_SHIFTS = ((0, 0), (1, 0), (0, 1), (1, 1))

# Per-axis factors of one level at frequencies θ: the 1-D filters and the 1-D
# polyphase factors ½ Σ_a e^{±j(θ+a)k} f(θ+a) of the coset shift k ∈ {0, 1}.
function _tilp_axis(fp::FilterPair, θ::Float64)
    h, hπ = centred_dtft(fp.h, θ), centred_dtft(fp.h, θ + π)
    g, gπ = centred_dtft(fp.g, θ), centred_dtft(fp.g, θ + π)
    gpoly = ((g + gπ) / 2, cis(θ) * (g - gπ) / 2)       # k = 0, 1
    hpoly = ((h + hπ) / 2, cis(-θ) * (h - hπ) / 2)
    return h, g, gpoly, hpoly
end

"""
    _TILPMasks

The cumulative TILP masks of an `L`-level pyramid on one DFT grid: `ana[i][c]`
takes the input spectrum to channel `c` of level `i`, `syn[i][c]` takes that
channel back (the ¼ factors included), and `coarse_ana` / `coarse_syn` do the same
for the lowpass.  The input is recovered as `Σ syn·ana + coarse_syn·coarse_ana = 1`.
"""
struct _TILPMasks{M <: AbstractMatrix}
    ana::Vector{Vector{M}}
    syn::Vector{Vector{M}}
    coarse_ana::M
    coarse_syn::M
end

"""
    _tilp_masks(Tf, fp, dims, L; rows=1:dims[1]) -> _TILPMasks

TILP masks in `Complex{Tf}` on the `dims` grid, restricted to the frequency rows
`rows` (the `rfft` half-plane for real data).
"""
function _tilp_masks(
        ::Type{Tf}, fp::FilterPair, dims::NTuple{2, Int}, L::Int; rows = 1:dims[1]
    ) where {Tf <: AbstractFloat}
    n1, n2 = dims
    L >= 1 || throw(ArgumentError("a TILP needs at least one level (got $(L))"))
    T = ComplexF64
    pre_a = ones(T, length(rows), n2)       # Π_{j<i} H(2^(j-1) ω)
    pre_s = ones(T, length(rows), n2)       # Π_{j<i} G(2^(j-1) ω) / 4^(i-1)
    ana = Vector{Vector{Matrix{Complex{Tf}}}}(undef, L)
    syn = Vector{Vector{Matrix{Complex{Tf}}}}(undef, L)
    for i in 1:L
        s = 2^(i - 1)
        ax_r = [_tilp_axis(fp, 2π * s * (r - 1) / n1) for r in rows]
        ax_c = [_tilp_axis(fp, 2π * s * (c - 1) / n2) for c in 1:n2]
        θr = [2π * s * (r - 1) / n1 for r in rows]
        θc = [2π * s * (c - 1) / n2 for c in 1:n2]
        H = [a[1] * b[1] for a in ax_r, b in ax_c]
        G = [a[2] * b[2] for a in ax_r, b in ax_c]
        ana[i] = map(_TILP_SHIFTS) do (kr, kc)
            K = [
                cis(θr[p] * kr + θc[q] * kc) - ax_r[p][3][kr + 1] * ax_c[q][3][kc + 1] * H[p, q]
                    for p in eachindex(rows), q in 1:n2
            ]
            Matrix{Complex{Tf}}(pre_a .* K)
        end |> collect
        syn[i] = map(_TILP_SHIFTS) do (kr, kc)
            F = [
                cis(-(θr[p] * kr + θc[q] * kc)) - G[p, q] * ax_r[p][4][kr + 1] * ax_c[q][4][kc + 1]
                    for p in eachindex(rows), q in 1:n2
            ]
            Matrix{Complex{Tf}}(pre_s .* F ./ 4)
        end |> collect
        pre_a .*= H
        pre_s .*= G ./ 4
    end
    return _TILPMasks(ana, syn, Matrix{Complex{Tf}}(pre_a), Matrix{Complex{Tf}}(pre_s))
end

# Copy a host array into the array family of `ref` (identity on the host).
_like(ref::AbstractArray, A::Array) = copyto!(similar(ref, eltype(A), size(A)), A)
_like(::Array, A::Array) = A
_like(ref::SubArray, A::Array) = _like(parent(ref), A)

_adapt_masks(ref::AbstractArray, m::_TILPMasks) = _TILPMasks(
    [[_like(ref, a) for a in lvl] for lvl in m.ana],
    [[_like(ref, a) for a in lvl] for lvl in m.syn],
    _like(ref, m.coarse_ana), _like(ref, m.coarse_syn)
)
