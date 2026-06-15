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
# Only the final subbands are copied to the host `Matrix{T}` coefficient
# containers.  Each device kernel reproduces the CPU reduction order, so results
# match the host path (including the shift-invariant NSCT).

# Convert params to element type `T`, preserving the ladder filter.
function _params_as(::Type{T}, params::ContourletParams) where {T}
    fp = FilterPair{T}(T.(params.lp_filters.h), T.(params.lp_filters.g))
    qfp = Contourlets._convert_qfp(T, params.dfb_filters)
    return ContourletParams{T}(params.J, params.L_array, fp, qfp)
end

# ── Contourlet Transform ──────────────────────────────────────────────────────

"""
    ct_forward(image::AbstractGPUMatrix, params::ContourletParams) -> ContourletCoefficients

GPU Discrete Contourlet Transform forward pass.  Both the Laplacian-pyramid and
the directional filter-bank stages run on the device.  The returned coefficients
are host `Matrix` objects (matching the CPU `ct_forward` to Float32 precision).
"""
function ct_forward(image::AbstractGPUMatrix, params::ContourletParams)
    T_out = float(eltype(image))
    p = _params_as(T_out, params)
    fp, qfp = p.lp_filters, p.dfb_filters
    J, L = p.J, p.L_array
    img = T_out.(image)

    subbands = Vector{Vector{Matrix{T_out}}}(undef, J)
    coarse = img
    for j in 1:J
        coarse_j, bp_j = lp_decompose(coarse, fp)        # GPU
        dev_sb = dfb_decompose(bp_j, L[j], qfp)          # GPU DFB
        subbands[j] = [Array(s) for s in dev_sb]         # subbands → host
        coarse = coarse_j
    end
    return ContourletCoefficients(Array(coarse), subbands, p)
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
    T_out = float(eltype(image))
    p = _params_as(T_out, params)
    fp, qfp = p.lp_filters, p.dfb_filters
    J, L = p.J, p.L_array
    img = T_out.(image)

    subbands = Vector{Vector{Matrix{T_out}}}(undef, J)
    coarse = img
    for j in 1:J
        coarse_j, bp_j = nsp_decompose(coarse, fp, j)              # GPU
        dev_sb = nsdfb_decompose(bp_j, L[j], qfp, j)               # GPU NSDFB
        subbands[j] = [Array(s) for s in dev_sb]                   # subbands → host
        coarse = coarse_j
    end
    return NSCTCoefficients(Array(coarse), subbands, p)
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
