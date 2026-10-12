# Directional filter banks on the device.
#
# The DFB and NSDFB are pointwise products with precomputed equivalent filters on
# the DFT grid, alias folds and FFTs (see `directional/bank*.jl`), so a
# device run needs nothing beyond the bank's masks on the device and an FFT for the
# array type (cuFFT, rocFFT, …).  The masks are built on the host once and moved
# here; the buffers and plans of a `_BankRunner` are allocated with `similar` from a
# device array and planned with AbstractFFTs, so they land on the device as well.

function _bank_to_device(backend, bank::Contourlets._DirectionalBank)
    dev(xs) = [_to_device(backend, x) for x in xs]
    return Contourlets._DirectionalBank(
        dev(bank.ana), dev(bank.syn), dev(bank.phase_r), dev(bank.phase_c),
        bank.delay, bank.sampling, bank.dims, bank.levels, bank.half
    )
end

Contourlets._adapt_bank(ref::AbstractGPUArray, bank::Contourlets._DirectionalBank) =
    _bank_to_device(_gpu_backend(ref), bank)
Contourlets._adapt_bank(ref::GPUSubArray, bank::Contourlets._DirectionalBank) =
    _bank_to_device(_gpu_backend(ref), bank)
