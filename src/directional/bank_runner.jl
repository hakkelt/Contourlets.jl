# Applying a decimated directional bank: `_BankRunner` (a `_DirectionalBank` with
# its FFT buffers and plans), the alias fold / tiling on the DFT grid, and
# `_bank_analyze!` / `_bank_synthesize!`.

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
