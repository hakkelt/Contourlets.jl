# Glue between MultiscaleFilterBanks' mask-bank engine (`MaskBankEngine`, which
# runs the NSCT and TICT workspaces) and this package's stacked coefficients: an
# array of size `(n1, n2, n)` whose slice 1 is the lowpass and whose slices
# 2, …, n are the engine's outputs in item order.  `MaskItem.index` is
# (level, direction) for the NSCT and (level, channel, direction) for the TICT.

# Stacked coefficients `c` as the engine sees them: output (g, k) of group g
# (an NSCT level, or channel c of TICT level i as group 4(i - 1) + c) is slice
# `offsets[g] + k`.
struct _Stacked{A <: AbstractArray}
    c::A
    offsets::Vector{Int}
end

MultiscaleFilterBanks.mask_bank_subband(s::_Stacked, (j, k)::NTuple{2, Int}) = view(s.c, :, :, s.offsets[j] + k)
MultiscaleFilterBanks.mask_bank_subband(s::_Stacked, (i, c, k)::NTuple{3, Int}) = view(s.c, :, :, s.offsets[4 * (i - 1) + c] + k)
MultiscaleFilterBanks.mask_bank_coarse(s::_Stacked) = view(s.c, :, :, 1)
# The engine's chunked (device) paths write a chunk of outputs through one batched
# FFT into these slices.
MultiscaleFilterBanks.mask_bank_storage(s::_Stacked) = view(s.c, :, :, 2:size(s.c, 3))

# Zero stacked coefficients: `n` subbands of size `dims` in an array like `ref`.
_zero_stack(ref::AbstractArray, ::Type{Td}, dims::NTuple{2, Int}, n::Int) where {Td} =
    fill!(similar(ref, Td, (dims..., n)), zero(Td))

# Whether the slices of stacked host coefficients of element type `Td` and size
# `size(x)` can start off FFTW's 16-byte alignment, so that the plans that write
# them must be planned `UNALIGNED`.
_unaligned_slices(x::Array) = (length(x) * sizeof(eltype(x))) % 16 != 0
_unaligned_slices(::AbstractArray) = false

# The engine of a workspace for data like `x`, with the spectrum buffer `X` and
# the plans of `_ns_unnormalised_plans`.  The analysis folds the inverse FFT's
# 1/N into its masked products, which on a device saves a kernel per subband.
# A device engine also runs its analysis in chunks of subbands, each one batched
# FFT (`_ns_batch_plan`); a host engine threads over the subbands instead.  With
# `tunable`, a device engine can chunk its synthesis too (`_ns_forward_batch_plan`);
# that is only offered to `mask_bank_tune`, which keeps it where it measures
# faster (small grids) and the per-subband synthesis loop elsewhere.
function _mask_bank_engine(x::AbstractMatrix, X, pfwd, pbwd, pyramid_ana, pyramid_syn, coarse_ana, coarse_syn, banks, items; tunable::Bool = false)
    s = _ns_scale(x, X)
    batched = if x isa Array
        (;)
    elseif tunable
        (; planner = _ns_batch_plan, forward_planner = _ns_forward_batch_plan, output = x)
    else
        (; planner = _ns_batch_plan, output = x)
    end
    return MaskBankEngine(
        [MaskBank(b.ana, b.syn) for b in banks], items, X;
        pfwd, pinv = s * pbwd, pyramid = MaskBank(pyramid_ana, pyramid_syn), lowpass = (coarse_ana, coarse_syn),
        pbwd = pbwd, scale = s, batched...
    )
end

# The unnormalised inverse FFT of a stack of spectra `B` to a stack of outputs `R`.
_ns_batch_plan(B::AbstractArray, R::AbstractArray) = _ns_inverse_plan(B, R)

# The forward FFT of a stack of outputs `R` to a stack of spectra like `B`.
_ns_forward_batch_plan(R::AbstractArray{<:Real}, B::AbstractArray) = plan_rfft(R, 1:2)
_ns_forward_batch_plan(R::AbstractArray, B::AbstractArray) = plan_fft(R, 1:2)

# The engine `e` tuned on this device (`mask_bank_tune`) when `tune` is set,
# timed on zero data through the coefficient set `coeffs`; `e` otherwise.
_maybe_tune(e, tune::Bool, coeffs, x::AbstractArray) = tune ? mask_bank_tune(e, coeffs, fill!(x, zero(eltype(x)))) : e
# A host engine threads over its subbands and has nothing to tune; this method
# also keeps the host workspace's type independent of `tune`.
_maybe_tune(e, ::Bool, coeffs, ::Array) = e
