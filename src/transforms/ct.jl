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
    ct_forward(image, params::ContourletParams; workspace=nothing, threading=Auto()) -> ContourletCoefficients

Discrete Contourlet Transform forward pass.

Pass a preallocated `ContourletWorkspace` via `workspace` to reuse scratch
buffers across calls (allocation-free pyramid stage).

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
        workspace::Union{ContourletWorkspace, Nothing} = nothing,
        threading::ThreadingPolicy = Auto()
    )
    if workspace !== nothing
        Tw = eltype(workspace)
        coeffs = similar_coefficients(params, size(image); Td = Tw)
        img = Tw === eltype(image) ? image : Tw.(image)
        return ct_forward!(coeffs, img, params; workspace = workspace, threading = threading)
    end
    # No-workspace: direct construction keeps the return type concrete (type-stable).
    Td = _data_eltype(image)
    Tf = _filter_eltype(Td)
    img = Td === eltype(image) ? image : Td.(image)
    J = params.J
    L = params.L_array
    p_out = _convert_params(Tf, params)
    fp, qfp = p_out.lp_filters, p_out.dfb_filters
    subbands = Vector{Vector{Matrix{Td}}}(undef, J)
    coarse = img
    for j in 1:J
        coarse_j, bp_j = lp_decompose(coarse, fp)
        subbands[j] = dfb_decompose(bp_j, L[j], qfp; threading = threading)
        coarse = coarse_j
    end
    return ContourletCoefficients(coarse, subbands)
end

"""
    ct_forward!(coeffs::ContourletCoefficients, image, params::ContourletParams; workspace=nothing, threading=Auto()) -> coeffs

In-place CT forward pass writing into the preallocated `coeffs`.

Pass a `ContourletWorkspace` via `workspace` for an allocation-free pyramid
stage (the directional stage still allocates its tree when no workspace is
given).  Allocate `coeffs` with [`similar_coefficients`](@ref).
"""
function ct_forward!(
        coeffs::ContourletCoefficients,
        image::AbstractMatrix,
        params::ContourletParams;
        workspace::Union{ContourletWorkspace, Nothing} = nothing,
        threading::ThreadingPolicy = Auto()
    )
    workspace === nothing && return _ct_forward_noworkspace!(coeffs, image, params; threading = threading)
    return _ct_forward_ws_barrier!(coeffs, image, workspace, threading)
end

# Typed dispatch barrier — forces specialization on the concrete workspace type,
# avoiding Union-boxing overhead on the hot workspace path.
function _ct_forward_ws_barrier!(
        coeffs::ContourletCoefficients,
        image::AbstractMatrix,
        ws::ContourletWorkspace,
        threading::ThreadingPolicy
    )
    T = eltype(ws)
    img = T === eltype(image) ? image : T.(image)
    return _ct_forward_ws!(coeffs, img, ws; threading = threading)
end

# Internal dispatch seam — overloaded by ContourletsCUDAExt for graph capture.
function _ct_forward_ws!(
        coeffs::ContourletCoefficients{T},
        image::AbstractMatrix,
        ws::ContourletWorkspace{T};
        threading::ThreadingPolicy = Auto()
    ) where {T}
    _ct_forward_inner!(coeffs, image, ws; threading = threading)
    return coeffs
end

# No-workspace in-place path: mirrors the allocating body but writes into coeffs.
function _ct_forward_noworkspace!(
        coeffs::ContourletCoefficients,
        image::AbstractMatrix,
        params::ContourletParams;
        threading::ThreadingPolicy = Auto()
    )
    Td = eltype(coeffs.coarse)
    Tf = _filter_eltype(Td)
    img = Td === eltype(image) ? image : Td.(image)
    J = params.J
    L = params.L_array
    p_out = _convert_params(Tf, params)
    fp, qfp = p_out.lp_filters, p_out.dfb_filters
    coarse = img
    for j in 1:J
        coarse_j, bp_j = lp_decompose(coarse, fp)
        sbs = dfb_decompose(bp_j, L[j], qfp; threading = threading)
        for (k, sb) in enumerate(sbs)
            copyto!(coeffs.subbands[j][k], sb)
        end
        coarse = coarse_j
    end
    copyto!(coeffs.coarse, coarse)
    return coeffs
end

function _ct_forward_inner!(
        coeffs::ContourletCoefficients{T},
        image::AbstractMatrix,
        ws::ContourletWorkspace{T};
        threading::ThreadingPolicy = Auto()
    ) where {T}
    params = ws.params
    J = params.J
    L = params.L_array
    fp = params.lp_filters
    qfp = params.dfb_filters

    img = T === eltype(image) ? image : T.(image)
    _arena_reset!(ws.fwd_scratch)
    return _with_arena(ws.fwd_scratch) do
        current = img
        for j in 1:J
            n1, n2 = size(current)
            coarse_j = _scratch_like(current, cld(n1, 2), cld(n2, 2))
            bp_j = _scratch_like(current, n1, n2)
            tmp_j = _scratch_like(current, n1, n2)
            tmp2_j = _scratch_like(current, n1, n2)

            lp_decompose!(coarse_j, bp_j, current, fp; tmp = tmp_j, tmp2 = tmp2_j)
            sbs = dfb_decompose(bp_j, L[j], qfp; filter = ws.dfb_f_cache, threading = threading)   # DFB transients use the arena
            for (k, sb) in enumerate(sbs)
                copyto!(coeffs.subbands[j][k], sb)
            end
            current = coarse_j
        end
        copyto!(coeffs.coarse, current)
    end
end

"""
    ct_inverse(coeffs::ContourletCoefficients, params::ContourletParams; workspace=nothing, threading=Auto()) -> Matrix

Discrete Contourlet Transform inverse pass.

Pass a preallocated `ContourletWorkspace` via `workspace` to reuse scratch
buffers across calls.

# Examples
```jldoctest
julia> using Contourlets, Random

julia> x = randn(Xoshiro(1), 64, 64);

julia> p = ContourletParams(J = 2, L_array = [1, 2]);

julia> rec = ct_inverse(ct_forward(x, p), p);

julia> maximum(abs, rec .- x) < 1e-12
true
```
"""
function ct_inverse(
        coeffs::ContourletCoefficients,
        params::ContourletParams;
        workspace::Union{ContourletWorkspace, Nothing} = nothing,
        threading::ThreadingPolicy = Auto()
    )
    if workspace !== nothing
        Tw = eltype(workspace)
        M = _ws_matrix_type(workspace)
        image = _allocate_zeros(M, Tw, workspace.image_size)
        return ct_inverse!(image, coeffs, params; workspace = workspace, threading = threading)
    end
    return _ct_inverse_alloc(coeffs, params; threading = threading)
end

"""
    ct_inverse!(image, coeffs::ContourletCoefficients, params::ContourletParams; workspace=nothing, threading=Auto()) -> image

In-place CT inverse pass writing the reconstruction into `image`.

Pass a `ContourletWorkspace` via `workspace` for an allocation-free pyramid
stage.  The `image` array must match the size and element type of the original
input image (use `similar(original_image)` or `similar_coefficients` helpers).
"""
function ct_inverse!(
        image::AbstractMatrix,
        coeffs::ContourletCoefficients,
        params::ContourletParams;
        workspace::Union{ContourletWorkspace, Nothing} = nothing,
        threading::ThreadingPolicy = Auto()
    )
    workspace === nothing && return (copyto!(image, _ct_inverse_alloc(coeffs, params; threading = threading)); image)
    return _ct_inverse_ws_barrier!(image, coeffs, workspace, threading)
end

# Typed dispatch barrier — forces specialization on the concrete workspace type.
function _ct_inverse_ws_barrier!(
        image::AbstractMatrix,
        coeffs::ContourletCoefficients,
        ws::ContourletWorkspace,
        threading::ThreadingPolicy
    )
    return _ct_inverse_ws!(image, coeffs, ws; threading = threading)
end

# Internal dispatch seam — overloaded by ContourletsCUDAExt for graph capture.
function _ct_inverse_ws!(
        image::AbstractMatrix{T},
        coeffs::ContourletCoefficients{T},
        ws::ContourletWorkspace{T};
        threading::ThreadingPolicy = Auto()
    ) where {T}
    _ct_inverse_inner!(image, coeffs, ws; threading = threading)
    return image
end

# Allocating inverse — the reconstructed image size requires running the DFB,
# so we cannot pre-allocate from params alone without running the transform.
function _ct_inverse_alloc(
        coeffs::ContourletCoefficients,
        params::ContourletParams;
        threading::ThreadingPolicy = Auto()
    )
    Tf = _filter_eltype(eltype(coeffs.coarse))
    p_out = _convert_params(Tf, params)
    J = p_out.J
    fp = p_out.lp_filters
    qfp = p_out.dfb_filters
    coarse = copy(coeffs.coarse)

    for j in J:-1:1
        bp = dfb_reconstruct(coeffs.subbands[j], qfp; threading = threading)
        coarse = lp_reconstruct(coarse, bp, fp)
    end
    return coarse
end

function _ct_inverse_inner!(
        image::AbstractMatrix{T},
        coeffs::ContourletCoefficients{T},
        ws::ContourletWorkspace{T};
        threading::ThreadingPolicy = Auto()
    ) where {T}
    params = ws.params
    J = params.J
    fp = params.lp_filters
    qfp = params.dfb_filters

    _arena_reset!(ws.inv_scratch)
    _with_arena(ws.inv_scratch) do
        current = _scratch_like(image, size(coeffs.coarse, 1), size(coeffs.coarse, 2))
        copyto!(current, coeffs.coarse)

        for j in J:-1:1
            bp = dfb_reconstruct(coeffs.subbands[j], qfp; filter = ws.dfb_f_cache, threading = threading)   # DFB transients use the arena
            n1, n2 = size(bp)
            tmp_j = _scratch_like(current, n1, n2)
            tmp2_j = _scratch_like(current, n1, n2)
            if j > 1
                next_coarse = _scratch_like(current, n1, n2)
                lp_reconstruct!(next_coarse, current, bp, fp; tmp = tmp_j, tmp2 = tmp2_j)
                current = next_coarse
            else
                lp_reconstruct!(image, current, bp, fp; tmp = tmp_j, tmp2 = tmp2_j)
            end
        end
    end
    return image
end

# Extract the matrix storage type M from a ContourletWorkspace{Td,Tf,M,...}.
_ws_matrix_type(::ContourletWorkspace{Td, Tf, M}) where {Td, Tf, M} = M

# Grow the workspace scratch arena to its steady-state size by running one forward
# + inverse pass on zeros, so the first real call allocates nothing.  Called from
# `make_workspace(...; prewarm=true)`.
function _prewarm_ct!(ws::ContourletWorkspace{Td, Tf, M}) where {Td, Tf, M}
    img = _allocate_zeros(M, Td, ws.image_size)
    coeffs = similar_coefficients(ws.params, ws.image_size; Td = Td, M = M)
    ct_forward!(coeffs, img, ws.params; workspace = ws)
    ct_inverse!(img, coeffs, ws.params; workspace = ws)
    return ws
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
        Td::Type = Tf,
        M::Type{<:AbstractMatrix} = Matrix{Td}
    ) where {Tf}
    J = params.J
    L = params.L_array
    n1, n2 = image_size
    ladder = is_ladder(params.dfb_filters)
    A = typeof(_allocate_zeros(M, Td, (1, 1)))
    subbands = Vector{Vector{A}}(undef, J)
    cur_n1, cur_n2 = n1, n2
    for j in 1:J
        szs = dfb_subband_sizes(cur_n1, cur_n2, L[j]; ladder = ladder)
        subbands[j] = [_allocate_zeros(M, Td, s) for s in szs]
        cur_n1 = cld(cur_n1, 2)
        cur_n2 = cld(cur_n2, 2)
    end
    coarse = _allocate_zeros(M, Td, (cur_n1, cur_n2))
    return ContourletCoefficients(coarse, subbands)
end
