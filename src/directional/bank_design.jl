# Equivalent filters of the directional filter bank, built in closed form on the
# DFT grid.  The runtime bank built from them is in `bank.jl`; applying it (FFT
# buffers, alias folds) is in `bank_runner.jl`.
#
# Do (2001, thesis §3.3) shows that an l-level DFB is a multichannel filter bank:
# channel k filters with an equivalent filter Hₖ and downsamples by a *diagonal*
# matrix Sₖ (eq. 3.23),
#
#     yₖ[n] = (hₖ ⊛ x)[Sₖ n],   Sₖ = diag(2^(l-1), 2) or diag(2, 2^(l-1)),
#
# and that Hₖ is the product of the fan filters met along the tree, each evaluated
# at the frequency its node sees after the samplers above it (eq. 3.17):
#
#     Hₖ(ω) = K_{t₁t₂}(ω) · ∏_{i=2}^{l-1} F_{tᵢ,tᵢ₊₁}((Mₖ⁽ⁱ⁾)ᵀ ω).
#
# This file builds those Hₖ (and the synthesis Gₖ) as arrays over the DFT grid, so
# the decimated DFB becomes "multiply, fold the aliases, inverse FFT" per channel,
# and the nonsubsampled DFB of da Cunha, Zhou & Do (2006) — the same tree with the
# samplers removed and the filters upsampled accordingly — becomes "multiply,
# inverse FFT".  Filtering on the DFT grid is circular convolution, so both banks
# are exact on the torus: the NSDFB commutes with every circular shift and the DFB
# with every shift on the lattice of Sₖ.
#
# Coordinates.  The thesis writes ω = (ω₀, ω₁) with n₀ the horizontal index.  Here
# ω₀ is the frequency along the columns (dimension 2) and ω₁ along the rows
# (dimension 1), and every integer matrix below is in the thesis' (n₀, n₁) order.
#
# References:
#   M. N. Do, "Directional multiresolution image representations", PhD thesis,
#     EPFL, 2001, §3.3 (construction, eqs. 3.12–3.23).
#   S.-M. Phoong, C. W. Kim, P. P. Vaidyanathan & R. Ansari, "A new class of
#     two-channel biorthogonal filter banks and wavelet bases", IEEE Trans. SP
#     43(3), 1995, §V (the 2-D quincunx mapping, eqs. 28–32).
#   A. L. da Cunha, J. Zhou & M. N. Do, "The nonsubsampled contourlet transform:
#     theory, design, and applications", IEEE Trans. IP 15(10), 2006, §II-A.

# ── The quincunx fan filter pair (Phoong et al. 1995) ─────────────────────────
#
# Phoong et al. map a 1-D ladder filter bank to the quincunx lattice by replacing
# β(z) with β(z₀z₁⁻¹)β(z₀z₁) (eq. 32a):
#
#     H₀(z) = ½ (z₀^(-2N) + z₀⁻¹ β(z₀z₁⁻¹) β(z₀z₁)),
#     H₁(z) = −β(z₀z₁⁻¹) β(z₀z₁) H₀(z) + z₀^(-4N+1).
#
# With β the type-2 linear-phase filter β(z) = Σₖ vₖ (z^-(N-k) + z^-(N+k-1)),
# β(e^{jθ}) = e^{-j(N-½)θ} b(θ) where b(θ) = 2 Σₖ vₖ cos((k − ½)θ) is real.  The
# product β β carries the delay z₀^-(2N-1), so with p(ω) = b(ω₀+ω₁) b(ω₀−ω₁) and
# the common delay z₀^(-2N) dropped,
#
#     H₀ = (1 + p)/√2,    H₁ = e^{jω₀}(2 − p − p²)/√2,
#     G₀ = (2 + p − p²)/√2,    G₁ = e^{−jω₀}(1 − p)/√2,
#
# scaled by √2 so that ½(G₀H₀ + G₁H₁) = 1 and the aliasing term cancels, which is
# the quincunx perfect-reconstruction condition (thesis eqs. 3.9–3.10; p changes
# sign under ω ↦ ω + (π, π)).  This is the diamond pair; the fan pair the DFB uses
# is the same pair modulated by π in ω₀ (thesis §3.2.2), i.e. p is evaluated at
# (ω₀ + π, ω₁).  We label the fans as the thesis does: fan 0 is the modulated H₁,
# fan 1 the modulated H₀.
#
# Each fan filter is a real amplitude times a pure delay e^{jω·δ}: δ = (1, 0) for
# fan 0 and 0 for fan 1.  A product of fans evaluated at Bᵀω is again a real
# amplitude times e^{jω·d} with the integer delay d = Σ B δ, which is what lets
# the banks below store real masks plus one integer delay per channel.

# b(θ) for the half-filter v = (v₁, …, v_N) of a symmetric even-length ladder filter.
function _ladder_amplitude(v::AbstractVector{Float64}, θ::Float64)
    s = 0.0
    @inbounds for k in eachindex(v)
        s += v[k] * cos((k - 0.5) * θ)
    end
    return 2s
end

# The half v of the symmetric even-length ladder filter f = [reverse(v); v].
function _ladder_half(f::AbstractVector)
    n = length(f)
    (iseven(n) && n > 0) || throw(ArgumentError("the ladder filter must have even, nonzero length"))
    h = n ÷ 2
    all(f[h + i] == f[h + 1 - i] for i in 1:h) ||
        throw(ArgumentError("the ladder filter must be symmetric (Phoong et al.'s type-2 β)"))
    return Float64.(f[(h + 1):end])
end

# Fan amplitudes from p = b(ω₀+ω₁+π) b(ω₀−ω₁+π):
#   analysis  a₀ = −(2 − p − p²)/√2   a₁ = (1 + p)/√2
#   synthesis s₀ = −(1 − p)/√2        s₁ = (2 + p − p²)/√2
# (the minus signs are e^{jπ} from modulating the delay e^{jω₀}).
@inline _fan_analysis(f::Int, p) = f == 0 ? -(2 - p - p^2) / sqrt(2.0) : (1 + p) / sqrt(2.0)
@inline _fan_synthesis(f::Int, p) = f == 0 ? -(1 - p) / sqrt(2.0) : (2 + p - p^2) / sqrt(2.0)
const _FAN_DELAY = ((1, 0), (0, 0))

# ── Tree matrices (thesis eqs. 3.5–3.6 and 3.13) ──────────────────────────────

const _R = ([1 1; 0 1], [1 -1; 0 1])           # R₀, R₁ (R₂, R₃ are their transposes by J)
const _Q = ([1 -1; 1 1], [1 1; -1 1])          # Q₀, Q₁
const _J = [0 1; 1 0]                          # swaps n₀ and n₁
const _I2 = [1 0; 0 1]

# Resampled QFB of type τ ∈ 0:3 appended below a node with overall sampling M: its
# fan filters are seen at Bᵀω and the node's sampling becomes M·P_τ.  Types 0, 1
# serve the channels whose first fan is fan 0; types 2, 3 are the same banks with
# n₀ and n₁ swapped (thesis §3.3.1), for the other half.
function _qfb_type(τ::Int, M::Matrix{Int})
    if τ < 2
        R, Q = _R[τ + 1], _Q[τ + 1]
        return M * R, M * R * Q
    else
        R, Q = _R[τ - 1], _Q[τ - 1]
        return M * _J * R, M * _J * R * Q * _J
    end
end

# ── The tree on the DFT grid ──────────────────────────────────────────────────

# b on the DFT grid.  Every fan argument is θ = c·ω + π with an integer vector c, so
# on an n₁ × n₂ grid θ is a multiple of 2π/L (L = lcm(n₁, n₂)) plus π, and one
# table of b over its period 4π serves every factor of every channel.
struct _LadderTable
    tab::Vector{Float64}
    L::Int
    step_r::Int        # L ÷ n₁: table steps per unit of row frequency index
    step_c::Int        # L ÷ n₂
end

function _LadderTable(v::Vector{Float64}, n1::Int, n2::Int)
    L = lcm(n1, n2)
    tab = [_ladder_amplitude(v, 2π * m / L + π) for m in 0:(2L - 1)]
    return _LadderTable(tab, L, L ÷ n1, L ÷ n2)
end

# p(Bᵀω) = b(u₀ + u₁ + π) b(u₀ − u₁ + π) with u = Bᵀω, over the whole grid.
function _fan_argument!(p::Matrix{Float64}, t::_LadderTable, B::Matrix{Int})
    n1, n2 = size(p)
    # u₀ ± u₁ = (B[:,1] ± B[:,2])·ω with ω = (ω₀, ω₁) = 2π (k_c/n₂, k_r/n₁)
    c₊ = (B[1, 1] + B[1, 2], B[2, 1] + B[2, 2])
    c₋ = (B[1, 1] - B[1, 2], B[2, 1] - B[2, 2])
    P = 2 * t.L
    @inbounds for kc in 0:(n2 - 1), kr in 0:(n1 - 1)
        m₊ = mod(c₊[1] * kc * t.step_c + c₊[2] * kr * t.step_r, P)
        m₋ = mod(c₋[1] * kc * t.step_c + c₋[2] * kr * t.step_r, P)
        p[kr + 1, kc + 1] = t.tab[m₊ + 1] * t.tab[m₋ + 1]
    end
    return p
end

# One channel of the bank as built: real amplitudes of Hₖ and Gₖ on the grid, the
# integer delay dₖ (thesis order (n₀, n₁)) and the overall sampling matrix.
struct _BankChannel
    ana::Matrix{Float64}
    syn::Matrix{Float64}
    delay::Tuple{Int, Int}
    M::Matrix{Int}
    path::Vector{Int}          # fan digits t₁ t₂ … from the root
end

# Grow the subtree under `ch` to depth `l` (thesis §3.3.1), appending its leaves to
# `leaves` in order: level 2 is a quincunx bank on Q₀ for every node; from level 3
# a node reached through digit t is split by the resampled bank of type t (first
# fan 0) or 2 + t (first fan 1), and in the even types the fans swap places
# ("notice the ordering of frequency supports", Fig. 3.10).
#
# Only the leaves are kept: an inner node's amplitudes live in the scratch of its
# depth (`ana[d]`, `syn[d]`, and its fan argument in `p[d]`), so a depth-l tree
# allocates its 2^l leaves and l scratch levels, not all 2^(l+1) − 2 nodes.
function _grow!(
        leaves::Vector{_BankChannel}, ch::_BankChannel, t::_LadderTable, l::Int,
        p::Vector{Matrix{Float64}}, ana::Vector{Matrix{Float64}}, syn::Vector{Matrix{Float64}}
    )
    depth = length(ch.path)
    depth == l && return push!(leaves, ch)
    if depth == 1
        B, Mchild = _Q[1], [2 0; 0 2]
        fan_of_digit = identity
    else
        τ = ch.path[end] + 2 * ch.path[1]
        B, Mchild = _qfb_type(τ, ch.M)
        fan_of_digit = d -> iseven(τ) ? 1 - d : d
    end
    pd = _fan_argument!(p[depth], t, B)
    for digit in (0, 1)
        f = fan_of_digit(digit)
        δ = _FAN_DELAY[f + 1]
        d = (ch.delay[1] + B[1, 1] * δ[1] + B[1, 2] * δ[2], ch.delay[2] + B[2, 1] * δ[1] + B[2, 2] * δ[2])
        leaf = depth + 1 == l
        a = leaf ? similar(ch.ana) : ana[depth]
        s = leaf ? similar(ch.syn) : syn[depth]
        a .= ch.ana .* _fan_analysis.(f, pd)
        s .= ch.syn .* _fan_synthesis.(f, pd)
        _grow!(leaves, _BankChannel(a, s, d, Mchild, [ch.path; digit]), t, l, p, ana, syn)
    end
    return leaves
end

"""
    _bank_channels(n1, n2, l, f_ladder; half=:both) -> Vector{_BankChannel}

The `2^l` channels of an `l`-level DFB on an `n1 × n2` grid (or the `2^(l-1) + 1`
channels of a half tree), in output order.

Output order: the channels below fan 1 first, in increasing thesis index, then the
channels below fan 0 in decreasing thesis index — subband `k ≤ 2^(l-1)` is thesis
channel `2^(l-1) + k − 1` and subband `k > 2^(l-1)` is thesis channel `2^l − k`.
`half = :first` keeps only the fan-1 subtree split and appends the fan-0 node
unsplit; `half = :second` puts the unsplit fan-1 node first and splits fan 0
(the half trees of Eslami & Radha 2006).
"""
function _bank_channels(n1::Int, n2::Int, l::Int, f_ladder::AbstractVector; half::Symbol = :both)
    l >= 1 || throw(ArgumentError("a directional bank needs l ≥ 1"))
    t = _LadderTable(_ladder_half(f_ladder), n1, n2)
    p = Matrix{Float64}(undef, n1, n2)
    _fan_argument!(p, t, _I2)
    root(f) = _BankChannel(
        _fan_analysis.(f, p), _fan_synthesis.(f, p), _FAN_DELAY[f + 1], _I2, [f]
    )
    fan1, fan0 = root(1), root(0)
    scratch() = [Matrix{Float64}(undef, n1, n2) for _ in 1:l]
    ps, anas, syns = scratch(), scratch(), scratch()
    grow(ch) = _grow!(_BankChannel[], ch, t, l, ps, anas, syns)
    split1 = half === :second ? [fan1] : grow(fan1)
    split0 = half === :first ? [fan0] : reverse!(grow(fan0))
    return vcat(split1, split0)
end

# Diagonal sampling of a channel at depth `l ≥ 2` (thesis eq. 3.23, and its
# transpose for the other half), as Julia `(rows, cols)` factors.  Depth-1 channels
# are quincunx-sampled and marked `(0, 0)`.
function _channel_sampling(ch::_BankChannel)
    l = length(ch.path)
    l == 1 && return (0, 0)
    # fan-0 channels: diag(2^(l-1), 2) in (n₀, n₁) = (cols, rows); fan-1: transposed
    return ch.path[1] == 0 ? (2, 2^(l - 1)) : (2^(l - 1), 2)
end
