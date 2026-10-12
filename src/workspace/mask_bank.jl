# The mask-bank engine shared by the NSCT and TICT workspaces.
#
# Both transforms are pointwise products on the DFT grid: a pyramid stage given as
# masks on the input spectrum (`_nsp_masks` for the NSCT, `_tilp_masks` for the
# TICT) followed by a nonsubsampled directional bank (`_DirectionalBank`, the
# identity bank at depth 0).  With X the input spectrum, every output is
#
#     subband = ifft(bank.ana[k] · (pyramid_ana[g] · X)),    lowpass = ifft(coarse_ana · X),
#
# and the synthesis is
#
#     X = Σ_g pyramid_syn[g] · (Σ_k bank.syn[k] · fft(subband)) + coarse_syn · fft(lowpass).
#
# The items of one pyramid mask g are consecutive, so the bandpass spectrum
# pyramid_ana[g] · X is formed once per group and the synthesis applies
# pyramid_syn[g] once to the group's directional sum.  That costs a pass per group
# but saves a complex mask and a complex product per item: measured at 256², the
# elementwise work is ~35% (analysis) and ~45% (synthesis) cheaper than the
# fused per-item product for 16 items per group, and equal for 2.
#
# The outputs are independent, so both loops run in the threaded chunks of
# `fft_tasks.jl`: per task an FFT buffer and a group buffer and, for the synthesis
# from the second task on, a partial sum.  A chunk that starts inside a group
# forms that group's bandpass again, so the analysis gives the same bits whatever
# the chunking.

# One output: where it sits in the coefficient set (`index`, see `_subband`), the
# pyramid mask that feeds it and the channel of the bank that splits that mask.
struct _MaskItem{N}
    index::NTuple{N, Int}
    pyramid::Int
    bank::Int
    channel::Int
end

"""
    _MaskBankEngine

The precomputed masks, FFT plans and buffers of a nonsubsampled pyramid followed by
nonsubsampled directional banks, all on one DFT grid (the `rfft` half-plane for
real data).  `items` lists every output with its pyramid mask and bank channel;
`_mask_bank_forward!` / `_mask_bank_inverse!` run them.
"""
struct _MaskBankEngine{C <: AbstractMatrix, B, N, P, Q}
    pyramid_ana::Vector{C}      # pyramid masks, input spectrum → bandpass g
    pyramid_syn::Vector{C}      # and back
    coarse_ana::C               # input spectrum → lowpass
    coarse_syn::C
    banks::Vector{B}            # directional banks (shared between groups where equal)
    items::Vector{_MaskItem{N}} # grouped by pyramid mask
    X::C                        # input spectrum / synthesis accumulator
    buffers::Vector{C}          # per task: FFT buffer (consumed by the inverse FFT)
    groups::Vector{C}           # per task: bandpass spectrum / directional sum of a group
    partials::Vector{C}         # per task ≥ 2: partial synthesis sum
    pfwd::P                     # single-threaded plans between data and spectrum
    pinv::Q
end

# Build an engine for the spectrum buffer `X` and its plans from `_ns_plans`.
function _MaskBankEngine(
        pyramid_ana::Vector{C}, pyramid_syn::Vector{C}, coarse_ana::C, coarse_syn::C,
        banks::Vector{B}, items::Vector{_MaskItem{N}}, X::C, pfwd, pinv
    ) where {C, B, N}
    issorted(items; by = it -> it.pyramid) ||
        throw(ArgumentError("the items of one pyramid mask must be consecutive"))
    nt = _n_tasks(X, length(items))
    return _MaskBankEngine(
        pyramid_ana, pyramid_syn, coarse_ana, coarse_syn, banks, items, X,
        C[similar(X) for _ in 1:nt], C[similar(X) for _ in 1:nt], C[similar(X) for _ in 2:nt], pfwd, pinv
    )
end

_n_engine_tasks(e::_MaskBankEngine) = length(e.buffers)

# The output matrix of `index` in a coefficient set.
_subband(c::NSCTCoefficients, i::NTuple{2, Int}) = c.subbands[i[1]][i[2]]
_subband(c::TICTCoefficients, i::NTuple{3, Int}) = c.subbands[i[1]][i[2]][i[3]]

# Analysis of the dense input `x` into `coeffs` (subbands and lowpass).
function _mask_bank_forward!(coeffs, x::AbstractMatrix, e::_MaskBankEngine, threaded::Bool)
    mul!(e.X, e.pfwd, x)
    _batch_chunks(length(e.items), _n_engine_tasks(e), threaded) do range, t
        _mask_bank_forward_items!(coeffs, e, range, e.groups[t], e.buffers[t])
    end
    Z = e.buffers[1]
    Z .= e.coarse_ana .* e.X
    mul!(coeffs.coarse, e.pinv, Z)
    return coeffs
end

# The outputs `e.items[range]`, with the task's group buffer Y (the bandpass
# spectrum, formed when the group changes) and FFT buffer Z.
function _mask_bank_forward_items!(coeffs, e::_MaskBankEngine, range, Y, Z)
    g = 0
    for n in range
        it = e.items[n]
        if it.pyramid != g
            g = it.pyramid
            Y .= e.pyramid_ana[g] .* e.X
        end
        Z .= e.banks[it.bank].ana[it.channel] .* Y
        mul!(_subband(coeffs, it.index), e.pinv, Z)
    end
    return coeffs
end

# Synthesis of `coeffs` into the dense output `x`.
function _mask_bank_inverse!(x::AbstractMatrix, coeffs, e::_MaskBankEngine, threaded::Bool)
    acc = e.X
    fill!(acc, zero(eltype(acc)))
    _chunked_sum!(acc, e.partials, length(e.items), threaded; foreach = _batch_chunks) do a, range, t
        _mask_bank_inverse_items!(a, coeffs, e, range, e.buffers[t], e.groups[t])
    end
    F = e.buffers[1]
    mul!(F, e.pfwd, coeffs.coarse)
    acc .+= e.coarse_syn .* F
    mul!(x, e.pinv, acc)
    return x
end

# acc += the synthesis of the outputs `e.items[range]`, with the task's FFT buffer F
# and group buffer Y: a group's directional syntheses are summed in Y and its
# pyramid mask applied once.
function _mask_bank_inverse_items!(acc, coeffs, e::_MaskBankEngine, range, F, Y)
    g = 0
    for n in range
        it = e.items[n]
        if it.pyramid != g
            g == 0 || (acc .+= e.pyramid_syn[g] .* Y)
            fill!(Y, zero(eltype(Y)))
            g = it.pyramid
        end
        mul!(F, e.pfwd, _subband(coeffs, it.index))
        Y .+= e.banks[it.bank].syn[it.channel] .* F
    end
    g == 0 || (acc .+= e.pyramid_syn[g] .* Y)
    return acc
end
