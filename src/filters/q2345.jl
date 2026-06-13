# Quincunx filter pair for the Directional Filter Bank: the "23-45" biorthogonal
# filters of Phoong, Kim, Vaidyanathan & Ansari (1995), realised as a two-step
# ladder (lifting) network.
#
# Theory (Phoong et al. 1995; Do & Vetterli 2005, §V):
# The two-channel filter bank is parameterised by a single 1-D lifting filter
# β (the 12-tap "pkva" filter below).  After a shear operation the quincunx FB
# in this package reduces to a 1-D two-channel filter bank applied across one
# image axis.  The ladder realisation is
#
#   analysis:   y0 = (1/√2)·(p0 − B₁(p1))        p0/p1 = even/odd polyphase
#               y1 = −√2·p1 − B₀(y0)
#   synthesis:  p1 = (−1/√2)·(y1 + B₀(y0))
#               p0 = √2·y0 + B₁(p1)
#
# where B₀/B₁ filter with the modulated lifting filter f̃ (f̃ = β with odd taps
# negated, cf. fbdec_l.m in the reference MATLAB contourlet toolbox), and B₁
# uses an extra one-sample shift (half-sample alignment of the two phases).
#
# Perfect reconstruction is *structural*: the synthesis ladder inverts the
# analysis ladder step by step, exactly, for any lifting filter.
#
# The equivalent (non-ladder) filters have 23 taps (analysis low-pass h₀) and
# 45 taps (analysis high-pass h₁) — hence the name "23-45".  They are computed
# below by polyphase reassembly and stored in `h_q`/`g_q` for reference and for
# the nonsubsampled DFB, which filters at full rate with the equivalent filters.
#
# References:
#   Phoong, Kim, Vaidyanathan & Ansari (1995), "A new class of two-channel
#     biorthogonal filter banks and wavelet bases", IEEE Trans. SP 43(3).
#   Do & Vetterli (2005), "The contourlet transform", IEEE Trans. IP 14(12).

# 12-tap "pkva" ladder filter (half of the symmetric impulse response;
# identical to ldfilter('pkva12') in the reference MATLAB toolbox).
const _PKVA12_HALF = [0.63, -0.193, 0.0972, -0.0526, 0.0272, -0.0144]

"""
    pkva_ladder_filter() -> Vector{Float64}

The symmetric 12-tap "pkva" lifting filter of Phoong et al. (1995) used by the
default DFB filter pair [`Q2345`](@ref).
"""
pkva_ladder_filter() = [reverse(_PKVA12_HALF); _PKVA12_HALF]

# Modulate the lifting filter for the fan (directional) configuration:
# negate every odd (1-based) tap, cf. `f(1:2:end) = -f(1:2:end)` in fbdec_l.m.
function _ladder_modulate(f::AbstractVector{T}) where {T}
    return T[isodd(k) ? -f[k] : f[k] for k in eachindex(f)]
end

# ── Equivalent filters by polyphase reassembly ────────────────────────────────
#
# Lag-vector representation: a filter is `(taps::Vector, origin::Int)` where
# `taps[origin]` is the zero-lag coefficient, i.e. out[k] = Σ_m taps[m]·in[k − (m − origin)].

# Polynomial product of two lag vectors.
function _lag_conv(a::Vector{T}, ca::Int, b::Vector{T}, cb::Int) where {T}
    n = length(a) + length(b) - 1
    out = zeros(T, n)
    for i in eachindex(a), j in eachindex(b)
        out[i + j - 1] += a[i] * b[j]
    end
    return out, ca + cb - 1
end

# Accumulate `taps` (with origin `c`) into dict `acc` at lags 2·ℓ + off.
function _accum_up2!(acc::Dict{Int, T}, taps::Vector{T}, c::Int, off::Int) where {T}
    for m in eachindex(taps)
        lag = 2 * (m - c) + off
        acc[lag] = get(acc, lag, zero(T)) + taps[m]
    end
    return acc
end

function _dict_to_lagvec(acc::Dict{Int, T}) where {T}
    lo, hi = extrema(keys(acc))
    taps = zeros(T, hi - lo + 1)
    for (lag, v) in acc
        taps[lag - lo + 1] = v
    end
    return taps, 1 - lo
end

"""
    _ladder_equivalent_filters(f) -> NamedTuple

Compute the equivalent full-rate filters `(h0, c0, h1, c1, g0, cg0, g1, cg1)`
of the two-step ladder network with (already modulated) lifting filter `f`.
Channel 0 samples the odd image columns (1-based), channel 1 the even columns.

The filters satisfy `(g0 ⊛ h0 + g1 ⊛ h1) / 2 = δ` exactly (zero delay), which
is the perfect-reconstruction identity used by the nonsubsampled DFB.
"""
function _ladder_equivalent_filters(f::AbstractVector{T}) where {T}
    s2 = sqrt(T(2))
    c = length(f) ÷ 2                    # lifting filter origin
    fv = collect(T, f)
    # B₀: taps f at lags (m − c);  B₁: taps f at lags (m − c) + 1
    b0, cb0 = fv, c
    b1, cb1 = fv, c - 1                  # origin shifted ⇒ lags + 1

    acc = Dict{Int, T}()
    # h0 = (1/√2)·δ on the odd phase  −  (1/√2)·B₁ mapped to even-phase inputs
    _accum_up2!(acc, [one(T) / s2], 1, 0)
    _accum_up2!(acc, (-one(T) / s2) .* b1, cb1, -1)
    h0, c0 = _dict_to_lagvec(acc)

    # h1 = −√2·δ on the even phase  −  B₀ ∘ h0 (channel-0 feedback)
    acc = Dict{Int, T}()
    _accum_up2!(acc, [-s2], 1, 0)
    for m in eachindex(b0), d in eachindex(h0)
        lag = 2 * (m - cb0) + 1 + (d - c0)
        acc[lag] = get(acc, lag, zero(T)) - b0[m] * h0[d]
    end
    h1, c1 = _dict_to_lagvec(acc)

    # g1: odd phase −(1/√2)·B₁∘(−1/√2 ... ) — assemble from the synthesis ladder:
    #   p1 = (−1/√2)·y1 − (1/√2)·B₀(y0)
    #   p0 = √2·y0 + B₁(p1)
    # Contribution of y0 to x: odd phase √2·δ − (1/√2)·B₁B₀, even phase −(1/√2)·B₀
    bb, cbb = _lag_conv(b1, cb1, b0, cb0)
    acc = Dict{Int, T}()
    _accum_up2!(acc, [s2], 1, 0)
    _accum_up2!(acc, (-one(T) / s2) .* bb, cbb, 0)
    _accum_up2!(acc, (-one(T) / s2) .* b0, cb0, 1)
    g0, cg0 = _dict_to_lagvec(acc)

    # Contribution of y1 to x: odd phase −(1/√2)·B₁, even phase −(1/√2)·δ
    acc = Dict{Int, T}()
    _accum_up2!(acc, (-one(T) / s2) .* b1, cb1, -1)
    _accum_up2!(acc, [-one(T) / s2], 1, 0)
    g1, cg1 = _dict_to_lagvec(acc)

    return (h0 = h0, c0 = c0, h1 = h1, c1 = c1, g0 = g0, cg0 = cg0, g1 = g1, cg1 = cg1)
end

"""
    Q2345 :: QuincunxFilterPair{Float64}

The "23-45" biorthogonal filter pair of Phoong et al. (1995) for the DFB stage,
realised as a two-step ladder (lifting) network parameterised by the 12-tap
"pkva" filter (identical to the `'pkva'` filters of the reference MATLAB
contourlet toolbox).  Perfect reconstruction is structural — exact to machine
precision at every tree depth.

`h_q` and `g_q` hold the equivalent 23-tap analysis and 45-tap synthesis
low-pass filters; the ladder coefficient vector is stored in `f_ladder`.

Perfect-reconstruction property:

```jldoctest
julia> using Contourlets

julia> check_pr_condition(Q2345)
true
```
"""
const Q2345 = let
    f = pkva_ladder_filter()
    eq = _ladder_equivalent_filters(_ladder_modulate(f))
    QuincunxFilterPair{Float64}(
        reshape(eq.h0, 1, :), reshape(eq.g0, 1, :),
        (1, eq.c0), (1, eq.cg0), f
    )
end
