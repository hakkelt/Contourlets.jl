# Where the outputs of MultiscaleFilterBanks' mask-bank engine (`MaskBankEngine`,
# which runs the NSCT and TICT workspaces) live in this package's coefficient
# sets: `MaskItem.index` is (level, direction) for the NSCT and
# (level, channel, direction) for the TICT.

MultiscaleFilterBanks.mask_bank_subband(c::NSCTCoefficients, i::NTuple{2, Int}) = c.subbands[i[1]][i[2]]
MultiscaleFilterBanks.mask_bank_subband(c::TICTCoefficients, i::NTuple{3, Int}) = c.subbands[i[1]][i[2]][i[3]]

# The engine of a workspace for data like `x`, with the spectrum buffer `X` and
# the plans of `_ns_unnormalised_plans`.  The analysis folds the inverse FFT's
# 1/N into its masked products, which on a device saves a kernel per subband.
# A device engine also runs its analysis in chunks of subbands, each one batched
# FFT (`_ns_batch_plan`); a host engine threads over the subbands instead.
function _mask_bank_engine(x::AbstractMatrix, X, pfwd, pbwd, pyramid_ana, pyramid_syn, coarse_ana, coarse_syn, banks, items)
    s = _ns_scale(x, X)
    batched = x isa Array ? (;) : (; planner = _ns_batch_plan, output = x)
    return MaskBankEngine(
        [MaskBank(b.ana, b.syn) for b in banks], items, X, pfwd, s * pbwd;
        pyramid = MaskBank(pyramid_ana, pyramid_syn), lowpass = (coarse_ana, coarse_syn),
        pbwd = pbwd, scale = s, batched...
    )
end

# The unnormalised inverse FFT of a stack of spectra `B` to a stack of outputs `R`.
_ns_batch_plan(B::AbstractArray, R::AbstractArray) = _ns_inverse_plan(B, R)

# ── Subbands stored as one array ──────────────────────────────────────────────
#
# On a device the subbands of a coefficient set are the slices of one array (its
# `storage`), in engine order, so that a chunk's batched FFT writes them in place
# (`mask_bank_storage`).  This needs a slice to be an array of the subbands' own
# type, which holds for device arrays (a contiguous view of a `CuArray` is a
# `CuArray`) but not for `Array` (a `SubArray`): host subbands stay separate.

# `n` zero matrices of size `dims` and type like `M`, as the slices of one array
# when they can be, and that array (or `nothing`).
function _stacked_zeros(::Type{M}, ::Type{Td}, dims::NTuple{2, Int}, n::Int) where {M, Td}
    ref = _allocate_zeros(M, Td, (1, 1))
    A = typeof(ref)
    if !(ref isa Array)
        P = fill!(similar(ref, Td, (dims..., n)), zero(Td))
        slices = _slices_as(P, A)
        slices === nothing || return P, slices
    end
    return nothing, A[_allocate_zeros(M, Td, dims) for _ in 1:n]
end

# The slices `P[:, :, k]` as a `Vector{A}`, or `nothing` if a slice is not an `A`.
function _slices_as(P::AbstractArray{<:Any, 3}, ::Type{A}) where {A}
    slices = [view(P, :, :, k) for k in axes(P, 3)]
    return all(s -> s isa A, slices) ? A[s for s in slices] : nothing
end

# `flat` cut into consecutive vectors of `counts[i]` elements.
_split(flat::Vector, counts) = [flat[(e - c + 1):e] for (c, e) in zip(counts, cumsum(counts))]

MultiscaleFilterBanks.mask_bank_storage(c::Union{NSCTCoefficients, TICTCoefficients}) = c.storage
