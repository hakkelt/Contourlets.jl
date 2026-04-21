# Discrete Contourlet Transform (CT) — forward and inverse.
#
# Forward:
#   coarse ← image
#   for j = 1 … J:
#       coarse_j, bandpass_j = lp_decompose(coarse, lp_filters)
#       subbands[j] = dfb_decompose(bandpass_j, L_array[j], dfb_filters)
#       coarse = coarse_j
#
# Inverse (runs j = J … 1):
#       bandpass_j = dfb_reconstruct(subbands[j], dfb_filters)
#       coarse = lp_reconstruct(coarse, bandpass_j, lp_filters)

"""
    ct_forward(image, params::ContourletParams; workspace=nothing) -> ContourletCoefficients

Discrete Contourlet Transform forward pass.

Optionally pass a preallocated `ContourletWorkspace` to avoid per-call allocations.

# Examples
```jldoctest
julia> using Contourlets, Random; Random.seed!(1)
julia> x = randn(64, 64)
julia> p = ContourletParams(J=2, L_array=[1,2])
julia> coeffs = ct_forward(x, p)
julia> length(coeffs.subbands[2])
4
```
"""
function ct_forward(
        image::AbstractMatrix, params::ContourletParams{T};
        workspace::Union{ContourletWorkspace, Nothing} = nothing
    ) where {T}
    if workspace !== nothing
        coeffs = similar_coefficients(params, size(image))
        return ct_forward!(coeffs, image, workspace)
    end
    T_out = float(eltype(image))   # preserve image precision
    img = T_out.(image)
    J = params.J
    L = params.L_array
    fp = FilterPair{T_out}(T_out.(params.lp_filters.h), T_out.(params.lp_filters.g))
    qfp = QuincunxFilterPair{T_out}(
        T_out.(params.dfb_filters.h_q),
        T_out.(params.dfb_filters.g_q),
        params.dfb_filters.c_h,
        params.dfb_filters.c_g
    )
    subbands = Vector{Vector{Matrix{T_out}}}(undef, J)

    coarse = img
    for j in 1:J
        coarse_j, bp_j = lp_decompose(coarse, fp)
        subbands[j] = dfb_decompose(bp_j, L[j], qfp)
        coarse = coarse_j
    end
    p_out = ContourletParams{T_out}(J, L, fp, qfp)
    return ContourletCoefficients{T_out}(coarse, subbands, p_out)
end

"""
    ct_forward!(coeffs::ContourletCoefficients, image, ws::ContourletWorkspace) -> coeffs

In-place CT forward pass reusing preallocated workspace buffers.
"""
function ct_forward!(
        coeffs::ContourletCoefficients{T},
        image::AbstractMatrix,
        ws::ContourletWorkspace{T}
    ) where {T}
    params = coeffs.params
    J = params.J
    L = params.L_array
    fp = params.lp_filters
    qfp = params.dfb_filters

    # ws.current holds the full-size copy for level 1; levels 2..J use coarse_bufs[j-1]
    copyto!(ws.current, image)

    for j in 1:J
        input_j = (j == 1) ? ws.current : ws.coarse_bufs[j - 1]
        n1, n2 = size(input_j)
        tmp_j = @view ws.tmp_buf[1:n1, 1:n2]   # same-size scratch

        lp_decompose!(ws.coarse_bufs[j], ws.bp_bufs[j], input_j, fp; tmp = tmp_j)
        # DFB in-place would need per-level workspace; use allocating for now
        sbs = dfb_decompose(ws.bp_bufs[j], L[j], qfp)
        for (k, sb) in enumerate(sbs)
            copyto!(coeffs.subbands[j][k], sb)
        end
    end
    copyto!(coeffs.coarse, ws.coarse_bufs[J])
    return coeffs
end

"""
    ct_inverse(coeffs::ContourletCoefficients) -> Matrix

Discrete Contourlet Transform inverse pass.

# Examples
```jldoctest
julia> using Contourlets, Random; Random.seed!(1)
julia> x = randn(64, 64)
julia> p = ContourletParams(J=2, L_array=[1,2])
julia> rec = ct_inverse(ct_forward(x, p))
julia> maximum(abs, rec .- x) < 1e-12
true
```
"""
function ct_inverse(coeffs::ContourletCoefficients{T}) where {T}
    params = coeffs.params
    J = params.J
    fp = params.lp_filters
    qfp = params.dfb_filters
    coarse = copy(coeffs.coarse)

    for j in J:-1:1
        bp = dfb_reconstruct(coeffs.subbands[j], qfp)
        coarse = lp_reconstruct(coarse, bp, fp)
    end
    return coarse
end

"""
    ct_inverse!(image, coeffs::ContourletCoefficients, ws::ContourletWorkspace) -> image

In-place CT inverse pass.
"""
function ct_inverse!(
        image::AbstractMatrix{T},
        coeffs::ContourletCoefficients{T},
        ws::ContourletWorkspace{T}
    ) where {T}
    params = coeffs.params
    J = params.J
    fp = params.lp_filters
    qfp = params.dfb_filters

    # Seed coarse_bufs[J] from the stored coarse coefficients
    copyto!(ws.coarse_bufs[J], coeffs.coarse)

    for j in J:-1:1
        bp = dfb_reconstruct(coeffs.subbands[j], qfp)
        if j > 1
            # Output of reconstruction at level j is coarse input for level j-1.
            # coarse_bufs[j-1] has exactly that size: (n1/2^(j-1), n2/2^(j-1)).
            lp_reconstruct!(ws.coarse_bufs[j - 1], ws.coarse_bufs[j], bp, fp)
        else
            lp_reconstruct!(image, ws.coarse_bufs[j], bp, fp)
        end
    end
    return image
end

"""
    similar_coefficients(params::ContourletParams, image_size) -> ContourletCoefficients

Allocate a `ContourletCoefficients` structure with the correct buffer sizes for an
image of size `image_size`, without computing any transform.  Useful together with
`ct_forward!` in iterative algorithms.
"""
function similar_coefficients(
        params::ContourletParams{T},
        image_size::Tuple{Int, Int}
    ) where {T}
    J = params.J
    L = params.L_array
    n1, n2 = image_size
    subbands = Vector{Vector{Matrix{T}}}(undef, J)
    cur_n1, cur_n2 = n1, n2
    for j in 1:J
        n_dirs = max(1, 2^L[j])
        # Odd DFB levels split columns (n2 halved), even levels split rows (n1 halved)
        n_col_splits = cld(L[j], 2)      # ceil(L/2) col splits
        n_row_splits = L[j] ÷ 2         # floor(L/2) row splits
        sub_n1 = cur_n1 >> n_row_splits
        sub_n2 = cur_n2 >> n_col_splits
        subbands[j] = [zeros(T, sub_n1, sub_n2) for _ in 1:n_dirs]
        cur_n1 = cld(cur_n1, 2)
        cur_n2 = cld(cur_n2, 2)
    end
    coarse = zeros(T, cur_n1, cur_n2)
    return ContourletCoefficients{T}(coarse, subbands, params)
end
