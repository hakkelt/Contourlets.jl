# Nonsubsampled Contourlet Transform (NSCT) — forward and inverse.
#
# Same structure as the CT but with:
#   • lp_decompose → nsp_decompose (no downsampling, à trous upsampled filters)
#   • dfb_decompose → nsdfb_decompose (no quincunx decimation)
# All outputs have the same spatial size as the input.

"""
    nsct_forward(image, params::ContourletParams; workspace=nothing) -> NSCTCoefficients

Nonsubsampled Contourlet Transform forward pass.  Invariant under circular
shifts of the input.

Optionally pass a preallocated workspace from [`make_nsct_workspace`](@ref) to
reuse buffers across calls.

# Examples
```jldoctest
julia> using Contourlets, Random

julia> x = randn(Xoshiro(2), 32, 32);

julia> p = ContourletParams(J = 2, L_array = [1, 2]);

julia> coeffs = nsct_forward(x, p);

julia> size(coeffs.coarse) == size(x)
true
```
"""
function nsct_forward(
        image::AbstractMatrix, params::ContourletParams;
        workspace::Union{ContourletWorkspace, Nothing} = nothing
    )
    if workspace !== nothing
        Tw = eltype(workspace)   # the workspace dictates the data type
        coeffs = similar_nsct_coefficients(params, size(image); Td = Tw)
        img = Tw === eltype(image) ? image : Tw.(image)
        return nsct_forward!(coeffs, img, workspace)
    end
    Td = _data_eltype(image)       # data type (real or complex)
    Tf = _filter_eltype(Td)        # real filter precision
    img = Td === eltype(image) ? image : Td.(image)
    coarse = img
    J = params.J
    L = params.L_array
    p_out = _convert_params(Tf, params)   # real filters at precision Tf
    fp, qfp = p_out.lp_filters, p_out.dfb_filters
    subbands = Vector{Vector{Matrix{Td}}}(undef, J)

    for j in 1:J
        coarse_j, bp_j = nsp_decompose(coarse, fp, j)
        subbands[j] = nsdfb_decompose(bp_j, L[j], qfp, j)
        coarse = coarse_j
    end
    return NSCTCoefficients(coarse, subbands, p_out)
end

"""
    nsct_forward!(coeffs::NSCTCoefficients, image, ws::ContourletWorkspace) -> coeffs

In-place NSCT forward pass.
"""
function nsct_forward!(
        coeffs::NSCTCoefficients{T},
        image::AbstractMatrix,
        ws::ContourletWorkspace{T, Tf}
    ) where {T, Tf}
    params = coeffs.params
    J = params.J
    L = params.L_array
    fp = params.lp_filters

    _arena_reset!(ws.fwd_scratch)
    _with_arena(ws.fwd_scratch) do
        current = image
        for j in 1:J
            n1, n2 = size(current)
            coarse_j = _scratch_like(current, n1, n2)
            bp_j = _scratch_like(current, n1, n2)
            tmp_j = _scratch_like(current, n1, n2)
            tmp2_j = _scratch_like(current, n1, n2)

            nsp_decompose!(
                coarse_j, bp_j, current, fp, j;
                tmp = tmp_j, tmp2 = tmp2_j,
                h_j = ws.lp_h_cache[j], g_j = ws.lp_g_cache[j]
            )

            if ws.fft_plan_fwd !== nothing
                mul!(ws.fft_spectrum_bp, ws.fft_plan_fwd, bp_j)
                H_leaves = ws.nsdfb_H_cache[j]
                plan_inv = ws.fft_plan_inv
                for k in 1:length(H_leaves)
                    ws.fft_buffer_c .= ws.fft_spectrum_bp .* H_leaves[k]
                    mul!(coeffs.subbands[j][k], plan_inv, ws.fft_buffer_c)
                end
            else
                _nsdfb_decompose_into!(coeffs.subbands[j], bp_j, L[j], ws.qup_cache[j], ws.fwd_scratch)
            end

            current = coarse_j
        end
        copyto!(coeffs.coarse, current)
    end
    return coeffs
end

"""
    nsct_inverse(coeffs::NSCTCoefficients) -> Matrix

NSCT inverse pass.

# Examples
```jldoctest
julia> using Contourlets, Random

julia> x = randn(Xoshiro(2), 32, 32);

julia> p = ContourletParams(J = 2, L_array = [1, 2]);

julia> rec = nsct_inverse(nsct_forward(x, p));

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
        ws::ContourletWorkspace{T, Tf}
    ) where {T, Tf}
    params = coeffs.params
    J = params.J
    fp = params.lp_filters

    _arena_reset!(ws.inv_scratch)
    _with_arena(ws.inv_scratch) do
        current = _scratch_like(image, size(coeffs.coarse, 1), size(coeffs.coarse, 2))
        copyto!(current, coeffs.coarse)

        for j in J:-1:1
            n1, n2 = size(current)
            bp = _scratch_like(current, n1, n2)

            if ws.fft_plan_fwd !== nothing
                G_leaves = ws.nsdfb_G_cache[j]
                plan_fwd = ws.fft_plan_fwd
                plan_inv = ws.fft_plan_inv
                fill!(ws.fft_spectrum_bp, zero(eltype(ws.fft_spectrum_bp)))
                for k in 1:length(G_leaves)
                    mul!(ws.fft_buffer_c, plan_fwd, coeffs.subbands[j][k])
                    ws.fft_spectrum_bp .+= ws.fft_buffer_c .* G_leaves[k]
                end
                mul!(bp, plan_inv, ws.fft_spectrum_bp)
            else
                _nsdfb_reconstruct_into!(bp, coeffs.subbands[j], ws.qup_cache[j], ws.inv_scratch)
            end

            tmp_j = _scratch_like(current, n1, n2)
            tmp2_j = _scratch_like(current, n1, n2)

            if j > 1
                next_coarse = _scratch_like(current, n1, n2)
                nsp_reconstruct!(
                    next_coarse, current, bp, fp, j;
                    tmp = tmp_j, tmp2 = tmp2_j, g_j = ws.lp_g_cache[j]
                )
                current = next_coarse
            else
                nsp_reconstruct!(
                    image, current, bp, fp, j;
                    tmp = tmp_j, tmp2 = tmp2_j, g_j = ws.lp_g_cache[j]
                )
            end
        end
    end
    return image
end

# Grow the workspace scratch arena to its steady-state size by running one forward
# + inverse pass on zeros (the per-call allocation sequence is deterministic, so
# afterwards the first real call allocates nothing).  Called from
# `make_nsct_workspace(...; prewarm=true)`.
function _prewarm_nsct!(ws::ContourletWorkspace{Td}) where {Td}
    img = zeros(Td, ws.image_size)
    coeffs = similar_nsct_coefficients(ws.params, ws.image_size; Td = Td)
    nsct_forward!(coeffs, img, ws)
    nsct_inverse!(img, coeffs, ws)
    return ws
end

"""
    similar_nsct_coefficients(params::ContourletParams, image_size) -> NSCTCoefficients

Allocate `NSCTCoefficients` buffers (all subbands same spatial size as image).
"""
function similar_nsct_coefficients(
        params::ContourletParams{Tf},
        image_size::Tuple{Int, Int};
        Td::Type = Tf
    ) where {Tf}
    J = params.J
    L = params.L_array
    n1, n2 = image_size
    subbands = Vector{Vector{Matrix{Td}}}(undef, J)
    for j in 1:J
        n_dirs = max(1, 2^L[j])
        subbands[j] = [zeros(Td, n1, n2) for _ in 1:n_dirs]
    end
    coarse = zeros(Td, n1, n2)
    return NSCTCoefficients(coarse, subbands, params)
end
