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
julia> using Contourlets, Random; Random.seed!(1);

julia> x = randn(64, 64);

julia> p = ContourletParams(J=2, L_array=[1,2]);

julia> coeffs = ct_forward(x, p);

julia> length(coeffs.subbands[2])
4
```
"""
function ct_forward(
        image::AbstractMatrix, params::ContourletParams;
        workspace::Union{ContourletWorkspace, Nothing} = nothing
    )
    if workspace !== nothing
        Tw = eltype(workspace.tmp_buf)   # the workspace dictates the data type
        coeffs = similar_coefficients(params, size(image); Td = Tw)
        img = Tw === eltype(image) ? image : Tw.(image)
        return ct_forward!(coeffs, img, workspace)
    end
    Td = _data_eltype(image)       # data type (real or complex)
    Tf = _filter_eltype(Td)        # real filter precision
    img = Td === eltype(image) ? image : Td.(image)
    J = params.J
    L = params.L_array
    p_out = _convert_params(Tf, params)   # real filters at precision Tf
    fp, qfp = p_out.lp_filters, p_out.dfb_filters
    subbands = Vector{Vector{Matrix{Td}}}(undef, J)

    coarse = img
    for j in 1:J
        coarse_j, bp_j = lp_decompose(coarse, fp)
        subbands[j] = dfb_decompose(bp_j, L[j], qfp)
        coarse = coarse_j
    end
    return ContourletCoefficients(coarse, subbands, p_out)
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
        tmp_j = @view ws.tmp_buf[1:n1, 1:n2]    # same-size scratch
        tmp2_j = @view ws.tmp_buf2[1:n1, 1:n2]

        lp_decompose!(ws.coarse_bufs[j], ws.bp_bufs[j], input_j, fp; tmp = tmp_j, tmp2 = tmp2_j)
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
julia> using Contourlets, Random

julia> x = randn(Xoshiro(1), 64, 64);

julia> p = ContourletParams(J = 2, L_array = [1, 2]);

julia> rec = ct_inverse(ct_forward(x, p));

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
        n1, n2 = size(bp)
        tmp_j = @view ws.tmp_buf[1:n1, 1:n2]
        tmp2_j = @view ws.tmp_buf2[1:n1, 1:n2]
        if j > 1
            # Output of reconstruction at level j is coarse input for level j-1.
            # coarse_bufs[j-1] has exactly that size: (n1/2^(j-1), n2/2^(j-1)).
            lp_reconstruct!(ws.coarse_bufs[j - 1], ws.coarse_bufs[j], bp, fp; tmp = tmp_j, tmp2 = tmp2_j)
        else
            lp_reconstruct!(image, ws.coarse_bufs[j], bp, fp; tmp = tmp_j, tmp2 = tmp2_j)
        end
    end
    return image
end

"""
    dfb_subband_sizes(n1, n2, l; ladder=true) -> Vector{Tuple{Int,Int}}

Per-subband `(rows, cols)` sizes produced by an `l`-level DFB applied to an
`n1 × n2` bandpass image.  For the ladder (default) DFB the first `2^(l-1)`
subbands form the horizontal mosaic and the remaining `2^(l-1)` the vertical
mosaic; for the modulation-mode DFB all subbands are equal-sized.
"""
function dfb_subband_sizes(n1::Int, n2::Int, l::Int; ladder::Bool = true)
    l <= 0 && return [(n1, n2)]
    if !ladder
        sub_n1 = n1 >> (l ÷ 2)
        sub_n2 = n2 >> cld(l, 2)
        return fill((sub_n1, sub_n2), 2^l)
    end
    l == 1 && return fill((n1 >> 1, n2), 2)
    half = 2^(l - 1)
    first_sz = (n1 >> (l - 1), n2 >> 1)   # horizontal mosaic
    second_sz = (n1 >> 1, n2 >> (l - 1))  # vertical mosaic
    return vcat(fill(first_sz, half), fill(second_sz, half))
end

"""
    similar_coefficients(params::ContourletParams, image_size) -> ContourletCoefficients

Allocate a `ContourletCoefficients` structure with the correct buffer sizes for an
image of size `image_size`, without computing any transform.  Useful together with
`ct_forward!` in iterative algorithms.
"""
function similar_coefficients(
        params::ContourletParams{Tf},
        image_size::Tuple{Int, Int};
        Td::Type = Tf
    ) where {Tf}
    J = params.J
    L = params.L_array
    n1, n2 = image_size
    ladder = is_ladder(params.dfb_filters)
    subbands = Vector{Vector{Matrix{Td}}}(undef, J)
    cur_n1, cur_n2 = n1, n2
    for j in 1:J
        szs = dfb_subband_sizes(cur_n1, cur_n2, L[j]; ladder = ladder)
        subbands[j] = [zeros(Td, s...) for s in szs]
        cur_n1 = cld(cur_n1, 2)
        cur_n2 = cld(cur_n2, 2)
    end
    coarse = zeros(Td, cur_n1, cur_n2)
    return ContourletCoefficients(coarse, subbands, params)
end
