# GPU overloads for the top-level CT and NSCT transforms.
#
# Design: the whole transform runs on the device.  The multiscale (Laplacian /
# nonsubsampled) pyramid stage dispatches to the GPU primitives, and both
# directional filter banks stay on the device too:
#
#  • CT  — the decimated DFB (`dfb_gpu.jl`): the recursive polyphase tree is
#    built from slicing / circshift / broadcast plus the GPU `_resamp` and
#    `_sefilter2` kernels.
#  • NSCT — the NSDFB (`nsdfb_gpu.jl`): a per-pixel directional convolution.
#
# Each device kernel reproduces the CPU reduction order, so results match the
# host path (including the shift-invariant NSCT).  Coefficients are returned with
# their arrays still on the device (the containers carry a generic
# `A <: AbstractMatrix` storage type), so nothing is implicitly transferred to the
# host; use `Array(·)` or `Adapt.adapt(Array, coeffs)` to bring them back.

# ── Contourlet Transform ──────────────────────────────────────────────────────

"""
    ct_forward(image::AbstractGPUMatrix, params::ContourletParams) -> ContourletCoefficients

GPU Discrete Contourlet Transform forward pass.  Both the Laplacian-pyramid and
the directional filter-bank stages run on the device, and the returned
coefficients keep their arrays on the device (matching the CPU `ct_forward` to
Float32 precision).  `ct_inverse(coeffs)` then reconstructs on the device too.
"""
function ct_forward(image::AbstractGPUMatrix, params::ContourletParams)
    Td = Contourlets._data_eltype(image)        # data type (real or complex)
    Tf = Contourlets._filter_eltype(Td)         # real filter precision
    p = Contourlets._convert_params(Tf, params)
    fp, qfp = p.lp_filters, p.dfb_filters
    J, L = p.J, p.L_array
    img = Td === eltype(image) ? image : Td.(image)

    local subbands
    coarse = img
    for j in 1:J
        coarse_j, bp_j = lp_decompose(coarse, fp)        # GPU
        dev_sb = dfb_decompose(bp_j, L[j], qfp)          # GPU DFB — stays on device
        j == 1 && (subbands = Vector{typeof(dev_sb)}(undef, J))
        subbands[j] = dev_sb
        coarse = coarse_j
    end
    return ContourletCoefficients(coarse, subbands, p)   # device-resident coeffs
end

"""
    ct_inverse(coeffs::ContourletCoefficients, device::AbstractGPUArray) -> GPU Matrix

Inverse Contourlet Transform on GPU.  Pass any device array `device` (e.g. the
input image used for the forward pass); its KernelAbstractions backend selects
the output device.  Both the directional reconstruction and the pyramid
synthesis run on the device.
"""
function ct_inverse(coeffs::ContourletCoefficients{T}, device::AbstractGPUArray) where {T}
    backend = _gpu_backend(device)
    J = coeffs.params.J
    fp = coeffs.params.lp_filters
    qfp = coeffs.params.dfb_filters
    coarse = _to_device(backend, coeffs.coarse)
    for j in J:-1:1
        dev_sbs = [_to_device(backend, s) for s in coeffs.subbands[j]]  # subbands → device
        bp = dfb_reconstruct(dev_sbs, qfp)                  # GPU DFB
        coarse = lp_reconstruct(coarse, bp, fp)             # GPU
    end
    return coarse
end

# ── Nonsubsampled Contourlet Transform ────────────────────────────────────────

"""
    nsct_forward(image::AbstractGPUMatrix, params::ContourletParams) -> NSCTCoefficients

GPU Nonsubsampled Contourlet Transform forward pass (GPU pyramid + GPU NSDFB).
"""
function nsct_forward(image::AbstractGPUMatrix, params::ContourletParams)
    Td = Contourlets._data_eltype(image)        # data type (real or complex)
    Tf = Contourlets._filter_eltype(Td)         # real filter precision
    p = Contourlets._convert_params(Tf, params)
    fp, qfp = p.lp_filters, p.dfb_filters
    J, L = p.J, p.L_array
    img = Td === eltype(image) ? image : Td.(image)

    local subbands
    coarse = img
    for j in 1:J
        coarse_j, bp_j = nsp_decompose(coarse, fp, j)              # GPU
        dev_sb = nsdfb_decompose(bp_j, L[j], qfp, j)               # GPU NSDFB — stays on device
        j == 1 && (subbands = Vector{typeof(dev_sb)}(undef, J))
        subbands[j] = dev_sb
        coarse = coarse_j
    end
    return NSCTCoefficients(coarse, subbands, p)   # device-resident coeffs
end

"""
    nsct_inverse(coeffs::NSCTCoefficients, device::AbstractGPUArray) -> GPU Matrix

Inverse NSCT on GPU.  Pass any device array `device`; its KernelAbstractions
backend selects the output device.
"""
function nsct_inverse(coeffs::NSCTCoefficients{T}, device::AbstractGPUArray) where {T}
    backend = _gpu_backend(device)
    J = coeffs.params.J
    fp = coeffs.params.lp_filters
    qfp = coeffs.params.dfb_filters
    coarse = _to_device(backend, coeffs.coarse)
    for j in J:-1:1
        dev_sbs = [_to_device(backend, s) for s in coeffs.subbands[j]]  # subbands → device
        bp = nsdfb_reconstruct(dev_sbs, qfp, j)                    # GPU NSDFB
        coarse = nsp_reconstruct(coarse, bp, fp, j)               # GPU
    end
    return coarse
end

# ── Moving whole coefficient sets between host and device ─────────────────────
# `Adapt.adapt(CuArray, coeffs)` moves the data arrays to the device and
# `Adapt.adapt(Array, coeffs)` brings them back; the real filters in `params`
# stay on the host either way.

function Adapt.adapt_structure(to, c::ContourletCoefficients)
    return ContourletCoefficients(
        Adapt.adapt(to, c.coarse),
        [[Adapt.adapt(to, s) for s in lvl] for lvl in c.subbands],
        c.params,
    )
end

function Adapt.adapt_structure(to, c::NSCTCoefficients)
    return NSCTCoefficients(
        Adapt.adapt(to, c.coarse),
        [[Adapt.adapt(to, s) for s in lvl] for lvl in c.subbands],
        c.params,
    )
end
