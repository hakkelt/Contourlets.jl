# Nonsubsampled Contourlet Transform (NSCT) — forward and inverse.
#
# Same structure as the CT but with:
#   • lp_decompose → nsp_decompose (no downsampling, à trous upsampled filters)
#   • dfb_decompose → nsdfb_decompose (no quincunx decimation)
# All outputs have the same spatial size as the input.

"""
    nsct_forward(image, params::ContourletParams; workspace=nothing, threading=Auto()) -> NSCTCoefficients

Nonsubsampled Contourlet Transform forward pass.  Invariant under circular
shifts of the input.

Pass a preallocated workspace from [`make_nsct_workspace`](@ref) via
`workspace` to reuse buffers across calls.

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
        workspace::Union{ContourletWorkspace, Nothing} = nothing,
        threading::ThreadingPolicy = Auto()
    )
    if workspace !== nothing
        Tw = eltype(workspace)
        coeffs = similar_nsct_coefficients(params, size(image); Td = Tw)
        img = Tw === eltype(image) ? image : Tw.(image)
        return nsct_forward!(coeffs, img, params; workspace = workspace, threading = threading)
    end
    # No-workspace: direct construction keeps the return type concrete (type-stable).
    Td = _data_eltype(image)
    Tf = _filter_eltype(Td)
    img = Td === eltype(image) ? image : Td.(image)
    coarse = img
    J = params.J
    L = params.L_array
    p_out = _convert_params(Tf, params)
    fp, qfp = p_out.lp_filters, p_out.dfb_filters
    subbands = Vector{Vector{Matrix{Td}}}(undef, J)
    for j in 1:J
        coarse_j, bp_j = nsp_decompose(coarse, fp, j)
        subbands[j] = nsdfb_decompose(bp_j, L[j], qfp, j; threading = threading)
        coarse = coarse_j
    end
    return NSCTCoefficients(coarse, subbands)
end

"""
    nsct_forward!(coeffs::NSCTCoefficients, image, params::ContourletParams; workspace=nothing, threading=Auto()) -> coeffs

In-place NSCT forward pass writing into the preallocated `coeffs`.

Pass a `ContourletWorkspace` via `workspace` for an allocation-free pyramid
stage.  Allocate `coeffs` with [`similar_nsct_coefficients`](@ref).

!!! note "FFT threading"
    The FFTW plans in `ws` are bound to the thread count chosen at workspace
    construction time.  Passing a `threading` kwarg that implies a different
    thread count will trigger a one-time warning; to change FFT threading,
    create a new workspace with the desired `threading` policy.
"""
function nsct_forward!(
        coeffs::NSCTCoefficients,
        image::AbstractMatrix,
        params::ContourletParams;
        workspace::Union{ContourletWorkspace, Nothing} = nothing,
        threading::ThreadingPolicy = Auto()
    )
    workspace === nothing && return _nsct_forward_noworkspace!(coeffs, image, params; threading = threading)
    return _nsct_forward_ws_barrier!(coeffs, image, workspace, threading)
end

# Typed dispatch barrier — forces specialization on the concrete workspace type,
# avoiding Union-boxing overhead on the hot workspace path.
function _nsct_forward_ws_barrier!(
        coeffs::NSCTCoefficients,
        image::AbstractMatrix,
        ws::ContourletWorkspace,
        threading::ThreadingPolicy
    )
    T = eltype(ws)
    img = T === eltype(image) ? image : T.(image)
    return _nsct_forward_ws!(coeffs, img, ws; threading = threading)
end

# Internal dispatch seam — overloaded by GPU extension if needed.
function _nsct_forward_ws!(
        coeffs::NSCTCoefficients{T},
        image::AbstractMatrix,
        ws::ContourletWorkspace{T, Tf};
        threading::ThreadingPolicy = Auto()
    ) where {T, Tf}
    _nsct_forward_inner!(coeffs, image, ws; threading = threading)
    return coeffs
end

# No-workspace in-place path: mirrors the allocating body but writes into coeffs.
function _nsct_forward_noworkspace!(
        coeffs::NSCTCoefficients,
        image::AbstractMatrix,
        params::ContourletParams;
        threading::ThreadingPolicy = Auto()
    )
    Td = eltype(coeffs.coarse)
    Tf = _filter_eltype(Td)
    img = Td === eltype(image) ? image : Td.(image)
    coarse = img
    J = params.J
    L = params.L_array
    p_out = _convert_params(Tf, params)
    fp, qfp = p_out.lp_filters, p_out.dfb_filters
    for j in 1:J
        coarse_j, bp_j = nsp_decompose(coarse, fp, j)
        sbs = nsdfb_decompose(bp_j, L[j], qfp, j; threading = threading)
        for (k, sb) in enumerate(sbs)
            copyto!(coeffs.subbands[j][k], sb)
        end
        coarse = coarse_j
    end
    copyto!(coeffs.coarse, coarse)
    return coeffs
end

function _nsct_forward_inner!(
        coeffs::NSCTCoefficients{T},
        image::AbstractMatrix,
        ws::ContourletWorkspace{T, Tf};
        threading::ThreadingPolicy = Auto()
    ) where {T, Tf}
    params = ws.params
    J = params.J
    L = params.L_array
    fp = params.lp_filters

    img = T === eltype(image) ? image : T.(image)
    if ws.fft_plan_fwd !== nothing && _use_threading(threading, T) != ws.fft_threaded
        @warn "threading kwarg conflicts with workspace FFT thread count; " *
            "FFT stage uses $(ws.fft_threaded ? "multi" : "single")-threaded plans " *
            "from workspace construction — create a new workspace to change FFT threading" maxlog = 1
    end
    _arena_reset!(ws.fwd_scratch)
    _with_arena(ws.fwd_scratch) do
        current = img
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

                # In the hybrid threading model, Polyester is used natively
                # inside the convolution kernels. For FFT-based convolution,
                # we just loop through the leaves.
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
    nsct_inverse(coeffs::NSCTCoefficients, params::ContourletParams; workspace=nothing, threading=Auto()) -> Matrix

NSCT inverse pass.

# Examples
```jldoctest
julia> using Contourlets, Random

julia> x = randn(Xoshiro(2), 32, 32);

julia> p = ContourletParams(J = 2, L_array = [1, 2]);

julia> rec = nsct_inverse(nsct_forward(x, p), p);

julia> maximum(abs, rec .- x) < 1e-12
true
```
"""
function nsct_inverse(
        coeffs::NSCTCoefficients,
        params::ContourletParams;
        workspace::Union{ContourletWorkspace, Nothing} = nothing,
        threading::ThreadingPolicy = Auto()
    )
    if workspace !== nothing
        Tw = eltype(workspace)
        M = _ws_matrix_type(workspace)
        image = _allocate_zeros(M, Tw, workspace.image_size)
        return nsct_inverse!(image, coeffs, params; workspace = workspace, threading = threading)
    end
    return _nsct_inverse_alloc(coeffs, params; threading = threading)
end

"""
    nsct_inverse!(image, coeffs::NSCTCoefficients, params::ContourletParams; workspace=nothing, threading=Auto()) -> image

In-place NSCT inverse pass writing the reconstruction into `image`.

Pass a `ContourletWorkspace` via `workspace` for an allocation-free pyramid
stage.

!!! note "FFT threading"
    See [`nsct_forward!`](@ref) for the FFT threading constraint.
"""
function nsct_inverse!(
        image::AbstractMatrix,
        coeffs::NSCTCoefficients,
        params::ContourletParams;
        workspace::Union{ContourletWorkspace, Nothing} = nothing,
        threading::ThreadingPolicy = Auto()
    )
    workspace === nothing && return (copyto!(image, _nsct_inverse_alloc(coeffs, params; threading = threading)); image)
    return _nsct_inverse_ws_barrier!(image, coeffs, workspace, threading)
end

# Typed dispatch barrier — forces specialization on the concrete workspace type.
function _nsct_inverse_ws_barrier!(
        image::AbstractMatrix,
        coeffs::NSCTCoefficients,
        ws::ContourletWorkspace,
        threading::ThreadingPolicy
    )
    return _nsct_inverse_ws!(image, coeffs, ws; threading = threading)
end

# Internal dispatch seam.
function _nsct_inverse_ws!(
        image::AbstractMatrix{T},
        coeffs::NSCTCoefficients{T},
        ws::ContourletWorkspace{T, Tf};
        threading::ThreadingPolicy = Auto()
    ) where {T, Tf}
    _nsct_inverse_inner!(image, coeffs, ws; threading = threading)
    return image
end

function _nsct_inverse_alloc(
        coeffs::NSCTCoefficients,
        params::ContourletParams;
        threading::ThreadingPolicy = Auto()
    )
    J = params.J
    Tf = _filter_eltype(eltype(coeffs.coarse))
    p_out = _convert_params(Tf, params)
    fp = p_out.lp_filters
    qfp = p_out.dfb_filters
    coarse = copy(coeffs.coarse)

    for j in J:-1:1
        bp = nsdfb_reconstruct(coeffs.subbands[j], qfp, j; threading = threading)
        coarse = nsp_reconstruct(coarse, bp, fp, j)
    end
    return coarse
end

function _nsct_inverse_inner!(
        image::AbstractMatrix{T},
        coeffs::NSCTCoefficients{T},
        ws::ContourletWorkspace{T, Tf};
        threading::ThreadingPolicy = Auto()
    ) where {T, Tf}
    params = ws.params
    J = params.J
    fp = params.lp_filters

    if ws.fft_plan_fwd !== nothing && _use_threading(threading, T) != ws.fft_threaded
        @warn "threading kwarg conflicts with workspace FFT thread count; " *
            "FFT stage uses $(ws.fft_threaded ? "multi" : "single")-threaded plans " *
            "from workspace construction — create a new workspace to change FFT threading" maxlog = 1
    end
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

                n_subbands = length(coeffs.subbands[j])

                # In the hybrid threading model, Polyester is used natively
                # inside the convolution kernels. For FFT-based convolution,
                # we just loop through the leaves.
                for k in 1:n_subbands
                    mul!(ws.fft_buffer_c, ws.fft_plan_fwd, coeffs.subbands[j][k])
                    ws.fft_spectrum_bp .+= ws.fft_buffer_c .* G_leaves[k]
                end
                mul!(bp, ws.fft_plan_inv, ws.fft_spectrum_bp)
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
function _prewarm_nsct!(ws::ContourletWorkspace{Td, Tf, M}; threading::ThreadingPolicy = Auto()) where {Td, Tf, M}
    img = _allocate_zeros(M, Td, ws.image_size)
    coeffs = similar_nsct_coefficients(ws.params, ws.image_size; Td = Td, M = M)
    nsct_forward!(coeffs, img, ws.params; workspace = ws, threading = threading)
    nsct_inverse!(img, coeffs, ws.params; workspace = ws, threading = threading)
    return ws
end

"""
    similar_nsct_coefficients(params::ContourletParams, image_size) -> NSCTCoefficients

Allocate `NSCTCoefficients` buffers (all subbands same spatial size as image).
"""
function similar_nsct_coefficients(
        params::ContourletParams{Tf},
        image_size::Tuple{Int, Int};
        Td::Type = Tf,
        M::Type{<:AbstractMatrix} = Matrix{Td}
    ) where {Tf}
    J = params.J
    L = params.L_array
    n1, n2 = image_size
    A = typeof(_allocate_zeros(M, Td, (1, 1)))
    subbands = Vector{Vector{A}}(undef, J)
    for j in 1:J
        n_dirs = max(1, 2^L[j])
        subbands[j] = [_allocate_zeros(M, Td, (n1, n2)) for _ in 1:n_dirs]
    end
    coarse = _allocate_zeros(M, Td, (n1, n2))
    return NSCTCoefficients(coarse, subbands)
end
