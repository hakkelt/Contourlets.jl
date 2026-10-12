# Equivalent filters of the directional filter bank, built in closed form on the
# DFT grid.
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

function _expand(ch::_BankChannel, t::_LadderTable, B::Matrix{Int}, Mchild::Matrix{Int}, fan_of_digit, p::Matrix{Float64})
    _fan_argument!(p, t, B)
    return map((0, 1)) do digit
        f = fan_of_digit(digit)
        δ = _FAN_DELAY[f + 1]
        d = (ch.delay[1] + B[1, 1] * δ[1] + B[1, 2] * δ[2], ch.delay[2] + B[2, 1] * δ[1] + B[2, 2] * δ[2])
        _BankChannel(
            ch.ana .* _fan_analysis.(f, p), ch.syn .* _fan_synthesis.(f, p),
            d, Mchild, [ch.path; digit]
        )
    end
end

# Grow the subtree under `ch` to depth `l` (thesis §3.3.1): level 2 is a quincunx
# bank on Q₀ for every node; from level 3 a node reached through digit t is split by
# the resampled bank of type t (first fan 0) or 2 + t (first fan 1), and in the even
# types the fans swap places ("notice the ordering of frequency supports", Fig. 3.10).
function _grow(ch::_BankChannel, t::_LadderTable, l::Int, p::Matrix{Float64})
    depth = length(ch.path)
    depth == l && return [ch]
    if depth == 1
        kids = _expand(ch, t, _Q[1], [2 0; 0 2], identity, p)
    else
        τ = ch.path[end] + 2 * ch.path[1]
        B, Mchild = _qfb_type(τ, ch.M)
        kids = _expand(ch, t, B, Mchild, d -> iseven(τ) ? 1 - d : d, p)
    end
    return vcat(_grow(kids[1], t, l, p), _grow(kids[2], t, l, p))
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
    split1 = half === :second ? [fan1] : _grow(fan1, t, l, p)
    split0 = half === :first ? [fan0] : reverse!(_grow(fan0, t, l, p))
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

# ── Runtime bank ──────────────────────────────────────────────────────────────

"""
    _DirectionalBank

The equivalent filters of one directional bank for one image size, ready to apply:
real amplitude masks in filter precision, stored as host or device arrays.

* Decimated (DFB): channel `k` computes `yₖ[n] = (hₖ ⊛ x)[Sₖn]` with
  `Hₖ(ω) = anaₖ(ω) e^{jω·dₖ}`.  The fold over the `|det Sₖ|` aliases is normalised
  inside `ana`, so synthesis is `x = Σₖ gₖ ⊛ (↑Sₖ yₖ)` with `Gₖ = synₖ e^{−jω·dₖ}`.
  Depth-1 channels (`l = 1`, or the unsplit fan of a half tree) are sampled on the
  quincunx lattice and stored column by column (see `_quincunx_pack!`).
* Nonsubsampled (NSDFB): `yₖ = hₖ ⊛ x` with the zero-phase `Hₖ = anaₖ`, both
  amplitudes scaled by `2^(-depth/2)` so that `Σₖ GₖHₖ = 1`.  The delays only move
  the decimation lattice, so a bank without decimation drops them and every
  subband stays registered with the input.  For real data the masks are cropped to
  the `rfft` half-plane.
"""
struct _DirectionalBank{A <: AbstractMatrix, V <: AbstractVector}
    ana::Vector{A}
    syn::Vector{A}
    phase_r::Vector{V}                  # e^{jω₁dₖ₁} along the rows (decimated only)
    phase_c::Vector{V}                  # e^{jω₀dₖ₀} along the columns
    delay::Vector{NTuple{2, Int}}       # (rows, cols) delay dₖ (decimated only), on the host
    sampling::Vector{NTuple{2, Int}}    # (rows, cols) decimation; (0, 0) quincunx; (1, 1) none
    dims::NTuple{2, Int}
    levels::Int
    half::Symbol
end

_is_decimated(b::_DirectionalBank) = !isempty(b.phase_r)

function _check_half(half::Symbol)
    half in (:both, :first, :second) ||
        throw(ArgumentError("half must be :both, :first or :second (got :$(half))"))
    return half
end

# Number of subbands of an `l`-level bank, full or half tree.
_bank_n_subbands(l::Int, half::Symbol) = l == 0 ? 1 : (half === :both ? 2^l : 2^(l - 1) + 1)

function _check_bank_dims(dims::NTuple{2, Int}, l::Int, decimated::Bool)
    decimated || return dims
    m = 2^max(l - 1, 1)
    all(n -> n % m == 0, dims) || throw(
        ArgumentError(
            "an $(l)-level DFB needs both image dimensions divisible by $(m) (got $(dims[1]) × $(dims[2]))"
        )
    )
    return dims
end

"""
    _directional_bank(Tf, dims, l, f_ladder; decimated, half=:both, real_data=false, upsample=0)

Build the equivalent filters of an `l`-level DFB (`decimated = true`) or NSDFB on a
`dims` grid from the ladder filter `f_ladder`, in precision `Tf`, as host arrays.
`real_data` crops nonsubsampled masks to the `rfft` half-plane.  `upsample = m`
replaces every nonsubsampled filter Hₖ(z) by Hₖ(z^(2^m I)) (da Cunha et al. 2006,
§II-A.3), which keeps the perfect-reconstruction identity Σₖ GₖHₖ = 1.
"""
function _directional_bank(
        ::Type{Tf}, dims::NTuple{2, Int}, l::Int, f_ladder::AbstractVector;
        decimated::Bool, half::Symbol = :both, real_data::Bool = false, upsample::Int = 0
    ) where {Tf <: AbstractFloat}
    _check_half(half)
    upsample >= 0 || throw(ArgumentError("upsample must be ≥ 0 (got $(upsample))"))
    decimated && upsample > 0 &&
        throw(ArgumentError("only the nonsubsampled bank takes upsampled filters"))
    _check_bank_dims(dims, l, decimated)
    n1, n2 = dims
    chs = _bank_channels(n1, n2, l, f_ladder; half = half)
    if decimated
        samp = map(_channel_sampling, chs)
        fold(s) = s == (0, 0) ? 1.0 : 1.0 / (s[1] * s[2])
        ana = [Tf.(fold(s) .* c.ana) for (c, s) in zip(chs, samp)]
        syn = [Tf.(c.syn) for c in chs]
        pr = [Complex{Tf}[cis(2π * c.delay[2] * k / n1) for k in 0:(n1 - 1)] for c in chs]
        pc = [Complex{Tf}[cis(2π * c.delay[1] * k / n2) for k in 0:(n2 - 1)] for c in chs]
        delay = [(c.delay[2], c.delay[1]) for c in chs]
    else
        # Hₖ(z^(2^m I)) has the frequency response Hₖ(2^m ω): sample the mask there.
        rows = mod.((2^upsample) .* (0:(real_data ? n1 ÷ 2 : n1 - 1)), n1) .+ 1
        cols = mod.((2^upsample) .* (0:(n2 - 1)), n2) .+ 1
        scale(c) = 2.0^(-length(c.path) / 2)
        ana = [Tf.(scale(c) .* c.ana[rows, cols]) for c in chs]
        syn = [Tf.(scale(c) .* c.syn[rows, cols]) for c in chs]
        pr = Vector{Complex{Tf}}[]
        pc = Vector{Complex{Tf}}[]
        delay = NTuple{2, Int}[]
        samp = fill((1, 1), length(chs))
    end
    return _DirectionalBank(ana, syn, pr, pc, delay, samp, dims, l, half)
end

# ── Quincunx storage of depth-1 channels ──────────────────────────────────────
#
# A depth-1 channel keeps the samples of the quincunx lattice {n₀ + n₁ even}.  They
# are stored column by column: column c of the n₁/2 × n₂ subband holds rows
# 0, 2, 4, … of an even column and rows −1, 1, 3, … of an odd one (0-based), i.e.
# y[r, c] = v[2r − (c mod 2), c].

function _quincunx_pack!(y::AbstractMatrix, V::AbstractMatrix)
    n1 = size(V, 1)
    m = n1 ÷ 2
    _store!(view(y, :, 1:2:size(y, 2)), view(V, 1:2:n1, 1:2:size(V, 2)))
    _store!(view(y, 2:m, 2:2:size(y, 2)), view(V, 2:2:(n1 - 2), 2:2:size(V, 2)))
    _store!(view(y, 1:1, 2:2:size(y, 2)), view(V, n1:n1, 2:2:size(V, 2)))
    return y
end

function _quincunx_unpack!(V::AbstractMatrix, y::AbstractMatrix)
    n1 = size(V, 1)
    m = n1 ÷ 2
    fill!(V, zero(eltype(V)))
    view(V, 1:2:n1, 1:2:size(V, 2)) .= view(y, :, 1:2:size(y, 2))
    view(V, 2:2:(n1 - 2), 2:2:size(V, 2)) .= view(y, 2:m, 2:2:size(y, 2))
    view(V, n1:n1, 2:2:size(V, 2)) .= view(y, 1:1, 2:2:size(y, 2))
    return V
end

# Write a complex intermediate into a subband of the data type.
_store!(y::AbstractArray{<:Real}, Y::AbstractArray) = (y .= real.(Y); y)
_store!(y::AbstractArray, Y::AbstractArray) = (y .= Y; y)

# ── Runner: one bank plus its FFT buffers and plans ───────────────────────────

"""
    _BankRunner

A `_DirectionalBank` together with the complex buffers and in-place FFT plans it
needs, so that repeated analyses and syntheses allocate nothing.  Built for host or
device arrays alike: every operation is a broadcast, a `sum!` or an FFT plan.
"""
struct _BankRunner{B <: _DirectionalBank, C <: AbstractMatrix, P, Q}
    bank::B
    X::C                     # full-size spectrum (or rfft half-plane, nonsubsampled real)
    Z::C                     # full-size per-channel scratch
    pfwd::P                  # in-place forward FFT on full-size buffers (decimated)
    pinv::Q                  # in-place inverse FFT on full-size buffers (decimated)
    sub::Vector{C}           # one buffer per distinct decimated subband size
    psub_fwd::Vector{P}
    psub_inv::Vector{Q}
    subidx::Vector{Int}      # channel → index into `sub` (0: quincunx)
end

# Distinct decimated subband sizes of a bank, and the size of each channel.
function _subband_sizes(b::_DirectionalBank)
    n1, n2 = b.dims
    return map(b.sampling) do s
        s == (0, 0) ? (n1 ÷ 2, n2) : (n1 ÷ s[1], n2 ÷ s[2])
    end
end

"""
    _bank_runner(bank, ref; flags=FFTW.ESTIMATE)

Buffers and plans for a *decimated* `bank`, allocated like the array `ref`
(host or device) with complex element type.
"""
function _bank_runner(bank::_DirectionalBank, ref::AbstractArray{<:Number}; flags = FFTW.ESTIMATE)
    _is_decimated(bank) || throw(ArgumentError("_bank_runner serves decimated banks"))
    Tc = complex(eltype(bank.ana[1]))
    X = fill!(similar(ref, Tc, bank.dims), zero(Tc))
    Z = similar(X)
    pfwd, pinv = _inplace_plans(X, flags)
    sizes = _subband_sizes(bank)
    distinct = unique(s for (s, smp) in zip(sizes, bank.sampling) if smp != (0, 0))
    sub = [fill!(similar(ref, Tc, s), zero(Tc)) for s in distinct]
    plans = [_inplace_plans(S, flags) for S in sub]
    subidx = Int[smp == (0, 0) ? 0 : something(findfirst(==(s), distinct), 0) for (s, smp) in zip(sizes, bank.sampling)]
    P, Q = typeof(pfwd), typeof(pinv)
    return _BankRunner(bank, X, Z, pfwd, pinv, sub, P[first(p) for p in plans], Q[last(p) for p in plans], subidx)
end

# In-place forward/inverse FFT plans for `A` (FFTW flags only apply to host arrays).
_inplace_plans(A::Array, flags) = (plan_fft!(A; flags = flags), plan_ifft!(A; flags = flags))
_inplace_plans(A::AbstractArray, flags) = (plan_fft!(A), plan_ifft!(A))

# The delay phase across the alias blocks.  On block (a, b) of the s₁ × s₂ tiling
# of the full grid, with m = n/s rows and columns per block,
#
#     e^{jω·d} at (r + a m₁, c + b m₂) = e^{jω·d} at (r, c) · e^{j2π(d₁a/s₁ + d₂b/s₂)},
#
# so the phase factors into the subband grid's phase and one constant per block.
# The fold and the tiling below apply the constant per block and the phase once,
# on the small grid, instead of a full-size phase product per channel.
_block_phase(::Type{Tc}, d::NTuple{2, Int}, s::NTuple{2, Int}, a::Int, b::Int) where {Tc} =
    Tc(cis(2π * (d[1] * a / s[1] + d[2] * b / s[2])))

# Y = e^{jω·d} Σ_blocks A·X: the |det S| aliases of the channel spectrum A·X summed
# into the subband spectrum Y.  Plain view broadcasts rather than `sum!`, so nothing
# is allocated on host or device (a requirement for CUDA graph capture of the
# workspace paths).
function _fold!(
        Y::AbstractMatrix, A::AbstractMatrix, X::AbstractMatrix, pr::AbstractVector, pc::AbstractVector,
        d::NTuple{2, Int}, s::NTuple{2, Int}
    )
    m1, m2 = size(Y)
    Tc = eltype(Y)
    Y .= view(A, 1:m1, 1:m2) .* view(X, 1:m1, 1:m2)
    for b in 0:(s[2] - 1), a in 0:(s[1] - 1)
        (a, b) == (0, 0) && continue
        rows = (a * m1 + 1):((a + 1) * m1)
        cols = (b * m2 + 1):((b + 1) * m2)
        Y .+= _block_phase(Tc, d, s, a, b) .* view(A, rows, cols) .* view(X, rows, cols)
    end
    Y .*= view(pr, 1:m1) .* transpose(view(pc, 1:m2))
    return Y
end

# X += G e^{−jω·d} · (spectrum of ↑S y): upsampling by S tiles the subband spectrum
# Y over the s₁ × s₂ blocks of the full grid.  Y is overwritten by e^{−jω·d} Y.
function _tile_add!(
        X::AbstractMatrix, G::AbstractMatrix, pr::AbstractVector, pc::AbstractVector, Y::AbstractMatrix,
        d::NTuple{2, Int}, s::NTuple{2, Int}
    )
    m1, m2 = size(Y)
    Tc = eltype(Y)
    Y .*= conj.(view(pr, 1:m1)) .* conj.(transpose(view(pc, 1:m2)))
    for b in 0:(s[2] - 1), a in 0:(s[1] - 1)
        rows = (a * m1 + 1):((a + 1) * m1)
        cols = (b * m2 + 1):((b + 1) * m2)
        view(X, rows, cols) .+= conj(_block_phase(Tc, d, s, a, b)) .* view(G, rows, cols) .* Y
    end
    return X
end

"""
    _bank_analyze!(out, x, r::_BankRunner) -> out

Decimated directional analysis of `x` into the preallocated subbands `out`.
"""
function _bank_analyze!(out::AbstractVector{<:AbstractMatrix}, x::AbstractMatrix, r::_BankRunner)
    r.X .= x
    r.pfwd * r.X
    return _bank_analyze_spectrum!(out, r)
end

# Analysis of the full spectrum already in `r.X` (left untouched).
function _bank_analyze_spectrum!(out::AbstractVector{<:AbstractMatrix}, r::_BankRunner)
    b = r.bank
    for k in eachindex(out)
        i = r.subidx[k]
        if i == 0
            r.Z .= b.ana[k] .* r.X .* b.phase_r[k] .* transpose(b.phase_c[k])
            r.pinv * r.Z
            _quincunx_pack!(out[k], r.Z)
        else
            Y = r.sub[i]
            _fold!(Y, b.ana[k], r.X, b.phase_r[k], b.phase_c[k], b.delay[k], b.sampling[k])
            r.psub_inv[i] * Y
            _store!(out[k], Y)
        end
    end
    return out
end

"""
    _bank_synthesize!(x, subbands, r::_BankRunner) -> x

Decimated directional synthesis: the inverse of [`_bank_analyze!`](@ref).
"""
function _bank_synthesize!(x::AbstractMatrix, subbands::AbstractVector{<:AbstractMatrix}, r::_BankRunner)
    _bank_synthesize_spectrum!(subbands, r)
    r.pinv * r.X
    return _store!(x, r.X)
end

# Synthesis up to the full spectrum, which is left in `r.X`.
function _bank_synthesize_spectrum!(subbands::AbstractVector{<:AbstractMatrix}, r::_BankRunner)
    b = r.bank
    fill!(r.X, zero(eltype(r.X)))
    for k in eachindex(subbands)
        i = r.subidx[k]
        if i == 0
            _quincunx_unpack!(r.Z, subbands[k])
            r.pfwd * r.Z
            r.X .+= b.syn[k] .* conj.(b.phase_r[k]) .* conj.(transpose(b.phase_c[k])) .* r.Z
        else
            Y = r.sub[i]
            Y .= subbands[k]
            r.psub_fwd[i] * Y
            _tile_add!(r.X, b.syn[k], b.phase_r[k], b.phase_c[k], Y, b.delay[k], b.sampling[k])
        end
    end
    return r.X
end
