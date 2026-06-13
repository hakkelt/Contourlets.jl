# GPU overloads for the top-level CT and NSCT transforms.
#
# Design: the multiscale (Laplacian / nonsubsampled) pyramid stage — which is
# dominated by separable convolutions — runs on the GPU, while the directional
# stage (DFB / NSDFB) runs on the host.  The directional stage is a recursive
# resampling/polyphase construction with data-dependent indexing that is not
# expressed as KernelAbstractions kernels; the coefficient containers also store
# plain CPU `Matrix{T}`, so the bandpass images are moved host-side for the
# directional split and the resulting subbands need no further transfer.  This
# keeps results bit-identical to the CPU path (including the ladder DFB and the
# shift-invariant NSCT) while still offloading the heavy convolutions.

# Convert params to element type `T`, preserving the ladder filter.
function _params_as(::Type{T}, params::ContourletParams) where {T}
    fp = FilterPair{T}(T.(params.lp_filters.h), T.(params.lp_filters.g))
    qfp = Contourlets._convert_qfp(T, params.dfb_filters)
    return ContourletParams{T}(params.J, params.L_array, fp, qfp)
end

# ── Contourlet Transform ──────────────────────────────────────────────────────

"""
    ct_forward(image::AbstractGPUMatrix, params::ContourletParams) -> ContourletCoefficients

GPU Discrete Contourlet Transform forward pass.  The Laplacian-pyramid stage runs
on the device; the directional filter bank runs on the host.  The returned
coefficients are host `Matrix` objects (bit-identical to the CPU `ct_forward`).
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
        subbands[j] = dfb_decompose(Array(bp_j), L[j], qfp)  # host directional
        coarse = coarse_j
    end
    return ContourletCoefficients{T_out}(Array(coarse), subbands, p)
end

"""
    ct_inverse(coeffs::ContourletCoefficients, backend) -> GPU Matrix

Inverse Contourlet Transform on GPU.  `backend` is a KernelAbstractions backend
(e.g. `KernelAbstractions.get_backend(x)` for a device array `x`).  The
directional reconstruction runs on the host; the pyramid synthesis on the device.
"""
function ct_inverse(coeffs::ContourletCoefficients{T}, backend) where {T}
    J = coeffs.params.J
    fp = coeffs.params.lp_filters
    qfp = coeffs.params.dfb_filters
    coarse = _to_device(backend, coeffs.coarse)
    for j in J:-1:1
        bp_cpu = dfb_reconstruct(coeffs.subbands[j], qfp)   # host directional
        bp = _to_device(backend, bp_cpu)
        coarse = lp_reconstruct(coarse, bp, fp)             # GPU
    end
    return coarse
end

# ── Nonsubsampled Contourlet Transform ────────────────────────────────────────

"""
    nsct_forward(image::AbstractGPUMatrix, params::ContourletParams) -> NSCTCoefficients

GPU Nonsubsampled Contourlet Transform forward pass (GPU pyramid + host NSDFB).
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
        subbands[j] = nsdfb_decompose(Array(bp_j), L[j], qfp, j)   # host directional
        coarse = coarse_j
    end
    return NSCTCoefficients{T_out}(Array(coarse), subbands, p)
end

"""
    nsct_inverse(coeffs::NSCTCoefficients, backend) -> GPU Matrix

Inverse NSCT on GPU.  `backend` is a KernelAbstractions backend.
"""
function nsct_inverse(coeffs::NSCTCoefficients{T}, backend) where {T}
    J = coeffs.params.J
    fp = coeffs.params.lp_filters
    qfp = coeffs.params.dfb_filters
    coarse = _to_device(backend, coeffs.coarse)
    for j in J:-1:1
        bp_cpu = nsdfb_reconstruct(coeffs.subbands[j], qfp, j)     # host directional
        bp = _to_device(backend, bp_cpu)
        coarse = nsp_reconstruct(coarse, bp, fp, j)               # GPU
    end
    return coarse
end
