# Nonsubsampled Contourlet Transform (NSCT) — forward and inverse.
#
# Same structure as the CT but with:
#   • lp_decompose → nsp_decompose (no downsampling, à trous upsampled filters)
#   • dfb_decompose → nsdfb_decompose (no quincunx decimation)
# All outputs have the same spatial size as the input.
#
# Reference: A. L. da Cunha, J. Zhou & M. N. Do, "The nonsubsampled contourlet
# transform: theory, design, and applications", IEEE TIP 15(10), 2006, §II-A.

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
        workspace::Union{NSCTWorkspace, Nothing} = nothing,
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
        subbands[j] = nsdfb_decompose(bp_j, L[j], qfp, j)
        coarse = coarse_j
    end
    return NSCTCoefficients(coarse, subbands)
end

"""
    nsct_forward!(coeffs::NSCTCoefficients, image, params::ContourletParams; workspace=nothing, threading=Auto()) -> coeffs

In-place NSCT forward pass writing into the preallocated `coeffs`.

Pass a `NSCTWorkspace` via `workspace` for an allocation-free pyramid
stage.  Allocate `coeffs` with [`similar_nsct_coefficients`](@ref).

!!! note "Threading"
    With a host workspace and threading on (`Enabled()`, or `Auto()` for complex
    data) the independent subband FFTs run in one task per Julia thread.
"""
function nsct_forward!(
        coeffs::NSCTCoefficients,
        image::AbstractMatrix,
        params::ContourletParams;
        workspace::Union{NSCTWorkspace, Nothing} = nothing,
        threading::ThreadingPolicy = Auto()
    )
    workspace === nothing && return _nsct_forward_noworkspace!(coeffs, image, params)
    return _nsct_forward_ws_barrier!(coeffs, image, workspace, threading)
end

# Typed dispatch barrier — forces specialization on the concrete workspace type,
# avoiding Union-boxing overhead on the hot workspace path.
function _nsct_forward_ws_barrier!(
        coeffs::NSCTCoefficients,
        image::AbstractMatrix,
        ws::NSCTWorkspace,
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
        ws::NSCTWorkspace{T, Tf};
        threading::ThreadingPolicy = Auto()
    ) where {T, Tf}
    _nsct_forward_inner!(coeffs, image, ws; threading = threading)
    return coeffs
end

# No-workspace in-place path: mirrors the allocating body but writes into coeffs.
function _nsct_forward_noworkspace!(
        coeffs::NSCTCoefficients,
        image::AbstractMatrix,
        params::ContourletParams
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
        sbs = nsdfb_decompose(bp_j, L[j], qfp, j)
        for (k, sb) in enumerate(sbs)
            copyto!(coeffs.subbands[j][k], sb)
        end
        coarse = coarse_j
    end
    copyto!(coeffs.coarse, coarse)
    return coeffs
end

# The workspace's FFT plans run on its dense matrix type; a view or other wrapper
# is staged through a grow-once scratch buffer of that type.
function _staging(a::ScratchArena, x::AbstractMatrix)
    _arena_reset!(a)
    return _arena_take!(a, x, size(x)...)
end
_dense_input(ws::NSCTWorkspace, x::AbstractMatrix) =
    x isa _ws_matrix_type(ws) ? x : copyto!(_staging(ws.fwd_scratch, x), x)

function _nsct_forward_inner!(
        coeffs::NSCTCoefficients{T},
        image::AbstractMatrix,
        ws::NSCTWorkspace{T, Tf};
        threading::ThreadingPolicy = Auto()
    ) where {T, Tf}
    img = _dense_input(ws, T === eltype(image) ? image : T.(image))
    # One FFT of the input; subband k of level j is ifft(Hₖ · Bⱼ · X) with the
    # precomputed directional filter Hₖ and pyramid mask Bⱼ (`_nsp_masks`), run by
    # the mask-bank engine (`mask_bank.jl`), threaded over the subbands.
    return _mask_bank_forward!(coeffs, img, ws.engine, _use_threading(threading, T))
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
        workspace::Union{NSCTWorkspace, Nothing} = nothing,
        threading::ThreadingPolicy = Auto()
    )
    if workspace !== nothing
        Tw = eltype(workspace)
        M = _ws_matrix_type(workspace)
        image = _allocate_zeros(M, Tw, workspace.image_size)
        return nsct_inverse!(image, coeffs, params; workspace = workspace, threading = threading)
    end
    return _nsct_inverse_alloc(coeffs, params)
end

"""
    nsct_inverse!(image, coeffs::NSCTCoefficients, params::ContourletParams; workspace=nothing, threading=Auto()) -> image

In-place NSCT inverse pass writing the reconstruction into `image`.

Pass a `NSCTWorkspace` via `workspace` for an allocation-free pyramid
stage.

!!! note "Threading"
    See [`nsct_forward!`](@ref): the subband FFTs split between threads the same way.
"""
function nsct_inverse!(
        image::AbstractMatrix,
        coeffs::NSCTCoefficients,
        params::ContourletParams;
        workspace::Union{NSCTWorkspace, Nothing} = nothing,
        threading::ThreadingPolicy = Auto()
    )
    workspace === nothing && return (copyto!(image, _nsct_inverse_alloc(coeffs, params)); image)
    return _nsct_inverse_ws_barrier!(image, coeffs, workspace, threading)
end

# Typed dispatch barrier — forces specialization on the concrete workspace type.
function _nsct_inverse_ws_barrier!(
        image::AbstractMatrix,
        coeffs::NSCTCoefficients,
        ws::NSCTWorkspace,
        threading::ThreadingPolicy
    )
    return _nsct_inverse_ws!(image, coeffs, ws; threading = threading)
end

# Internal dispatch seam.
function _nsct_inverse_ws!(
        image::AbstractMatrix{T},
        coeffs::NSCTCoefficients{T},
        ws::NSCTWorkspace{T, Tf};
        threading::ThreadingPolicy = Auto()
    ) where {T, Tf}
    _nsct_inverse_inner!(image, coeffs, ws; threading = threading)
    return image
end

function _nsct_inverse_alloc(
        coeffs::NSCTCoefficients,
        params::ContourletParams
    )
    J = params.J
    Tf = _filter_eltype(eltype(coeffs.coarse))
    p_out = _convert_params(Tf, params)
    fp = p_out.lp_filters
    qfp = p_out.dfb_filters
    coarse = copy(coeffs.coarse)

    for j in J:-1:1
        bp = nsdfb_reconstruct(coeffs.subbands[j], qfp, j)
        coarse = nsp_reconstruct(coarse, bp, fp, j)
    end
    return coarse
end

function _nsct_inverse_inner!(
        image::AbstractMatrix{T},
        coeffs::NSCTCoefficients{T},
        ws::NSCTWorkspace{T, Tf};
        threading::ThreadingPolicy = Auto()
    ) where {T, Tf}
    # image = ifft(Σⱼ Sⱼ Σₖ Gₖ · fft(subband k of level j) + S_{J+1} · fft(lowpass)),
    # with threading summed in one partial accumulator per task.
    threaded = _use_threading(threading, T)
    if image isa _ws_matrix_type(ws)
        _mask_bank_inverse!(image, coeffs, ws.engine, threaded)
    else
        copyto!(image, _mask_bank_inverse!(_staging(ws.inv_scratch, image), coeffs, ws.engine, threaded))
    end
    return image
end

# Run one forward + inverse pass on zeros, so the first real call is compiled and
# allocates nothing.  Called from `make_nsct_workspace(...; prewarm=true)`.
function _prewarm_nsct!(ws::NSCTWorkspace{Td, Tf, M}; threading::ThreadingPolicy = Auto()) where {Td, Tf, M}
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
