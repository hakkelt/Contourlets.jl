# Nonsubsampled Contourlet Transform (NSCT) — forward and inverse.
#
# Same structure as the CT but with:
#   • lp_decompose → nsp_decompose (no downsampling, à trous upsampled filters)
#   • dfb_decompose → nsdfb_decompose (no quincunx decimation)
# All outputs have the same spatial size as the input.

"""
    nsct_forward(image, params::ContourletParams) -> NSCTCoefficients

Nonsubsampled Contourlet Transform forward pass.  Fully shift-invariant.

# Examples
```jldoctest
julia> using Contourlets, Random; Random.seed!(2)
julia> x = randn(32, 32)
julia> p = ContourletParams(J=2, L_array=[1,2])
julia> coeffs = nsct_forward(x, p)
julia> size(coeffs.coarse) == size(x)
true
```
"""
function nsct_forward(image::AbstractMatrix, params::ContourletParams{T}) where {T}
    img = T.(image)
    coarse = img
    J = params.J
    L = params.L_array
    fp = params.lp_filters
    qfp = params.dfb_filters
    subbands = Vector{Vector{Matrix{T}}}(undef, J)

    for j in 1:J
        coarse_j, bp_j = nsp_decompose(coarse, fp, j)
        subbands[j] = nsdfb_decompose(bp_j, L[j], qfp, j)
        coarse = coarse_j
    end
    return NSCTCoefficients{T}(coarse, subbands, params)
end

"""
    nsct_forward!(coeffs::NSCTCoefficients, image, ws::ContourletWorkspace) -> coeffs

In-place NSCT forward pass.
"""
function nsct_forward!(
        coeffs::NSCTCoefficients{T},
        image::AbstractMatrix,
        ws::ContourletWorkspace{T}
    ) where {T}
    params = coeffs.params
    J = params.J
    L = params.L_array
    fp = params.lp_filters
    qfp = params.dfb_filters

    copyto!(ws.current, image)

    for j in 1:J
        nsp_decompose!(ws.current, ws.bp_bufs[j], ws.current, fp, j; tmp = ws.tmp_buf)
        sbs = nsdfb_decompose(ws.bp_bufs[j], L[j], qfp, j)
        for (k, sb) in enumerate(sbs)
            copyto!(coeffs.subbands[j][k], sb)
        end
        # ws.current now holds coarse_j (overwritten in-place by nsp_decompose!)
    end
    copyto!(coeffs.coarse, ws.current)
    return coeffs
end

"""
    nsct_inverse(coeffs::NSCTCoefficients) -> Matrix

NSCT inverse pass.

# Examples
```jldoctest
julia> using Contourlets, Random; Random.seed!(2)
julia> x = randn(32, 32)
julia> p = ContourletParams(J=2, L_array=[1,2])
julia> rec = nsct_inverse(nsct_forward(x, p))
julia> maximum(abs, rec .- x) < 1e-12
true
```
"""
function nsct_inverse(coeffs::NSCTCoefficients{T}) where {T}
    params = coeffs.params
    J = params.J
    fp = params.lp_filters
    qfp = params.dfb_filters
    coarse = copy(coeffs.coarse)

    for j in J:-1:1
        bp = nsdfb_reconstruct(coeffs.subbands[j], qfp, j)
        coarse = nsp_reconstruct(coarse, bp, fp, j)
    end
    return coarse
end

"""
    nsct_inverse!(image, coeffs::NSCTCoefficients, ws::ContourletWorkspace) -> image

In-place NSCT inverse pass.
"""
function nsct_inverse!(
        image::AbstractMatrix{T},
        coeffs::NSCTCoefficients{T},
        ws::ContourletWorkspace{T}
    ) where {T}
    params = coeffs.params
    J = params.J
    fp = params.lp_filters
    qfp = params.dfb_filters
    copyto!(ws.current, coeffs.coarse)

    for j in J:-1:1
        bp = nsdfb_reconstruct(coeffs.subbands[j], qfp, j)
        nsp_reconstruct!(image, ws.current, bp, fp, j; tmp = ws.tmp_buf)
        if j > 1
            copyto!(ws.current, image)
        end
    end
    return image
end

"""
    similar_nsct_coefficients(params::ContourletParams, image_size) -> NSCTCoefficients

Allocate `NSCTCoefficients` buffers (all subbands same spatial size as image).
"""
function similar_nsct_coefficients(
        params::ContourletParams{T},
        image_size::Tuple{Int, Int}
    ) where {T}
    J = params.J
    L = params.L_array
    n1, n2 = image_size
    subbands = Vector{Vector{Matrix{T}}}(undef, J)
    for j in 1:J
        n_dirs = max(1, 2^L[j])
        subbands[j] = [zeros(T, n1, n2) for _ in 1:n_dirs]
    end
    coarse = zeros(T, n1, n2)
    return NSCTCoefficients{T}(coarse, subbands, params)
end
