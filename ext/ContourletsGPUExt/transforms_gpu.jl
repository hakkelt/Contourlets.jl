# GPU overloads for the top-level CT and NSCT transforms.
#
# Design: the whole transform runs on the device.  The multiscale (Laplacian /
# nonsubsampled) pyramid stage dispatches to the GPU primitives, and both
# directional filter banks stay on the device too: they are products with
# precomputed equivalent filters, alias folds and FFTs, so they only need their
# masks copied to the device (`_adapt_bank`) and an array type with an FFT (cuFFT,
# rocFFT, …).
#
# Coefficients are returned with their arrays still on the device (the containers
# carry a generic `A <: AbstractMatrix` storage type), so nothing is implicitly
# transferred to the host; use `Array(·)` or `Adapt.adapt(Array, coeffs)` to bring
# them back.

# ── Contourlet Transform ──────────────────────────────────────────────────────

"""
    ct_forward(image::_AbstractGPUMatrix, params::ContourletParams) -> ContourletCoefficients

GPU Discrete Contourlet Transform forward pass.  Both the Laplacian-pyramid and
the directional filter-bank stages run on the device, and the returned
coefficients keep their arrays on the device (matching the CPU `ct_forward` to
Float32 precision).  `ct_inverse(coeffs, params)` then reconstructs on the device too.
"""
function Contourlets.ct_forward(
        image::_AbstractGPUMatrix, params::ContourletParams;
        workspace::Union{Contourlets.CTWorkspace, Nothing} = nothing
    )
    if workspace !== nothing
        Tw = eltype(workspace)
        M = Contourlets._ws_matrix_type(workspace)
        coeffs = Contourlets.similar_coefficients(params, size(image); Td = Tw, M = M)
        img = Tw === eltype(image) ? image : Tw.(image)
        return ct_forward!(coeffs, img, params; workspace = workspace)
    end
    Td = data_eltype(image)        # data type (real or complex)
    Tf = filter_eltype(Td)         # real filter precision
    p = Contourlets._convert_params(Tf, params)
    fp, qfp = p.lp_filters, p.dfb_filters
    J, L = p.J, p.L_array
    img = Td === eltype(image) ? image : Td.(image)

    local subbands
    coarse = img
    for j in 1:J
        coarse_j, bp_j = lp_decompose(coarse, fp)                    # GPU
        dev_sb = dfb_decompose(bp_j, L[j], qfp)                     # GPU DFB — stays on device
        j == 1 && (subbands = Vector{typeof(dev_sb)}(undef, J))
        subbands[j] = dev_sb
        coarse = coarse_j
    end
    return ContourletCoefficients(coarse, subbands)   # device-resident coeffs
end

"""
    ct_inverse(coeffs::ContourletCoefficients, params::ContourletParams) -> GPU matrix

GPU Discrete Contourlet Transform inverse pass.
"""
function Contourlets.ct_inverse(
        coeffs::ContourletCoefficients{T, <:AbstractGPUMatrix{T}},
        params::ContourletParams;
        workspace::Union{Contourlets.CTWorkspace, Nothing} = nothing
    ) where {T <: Number}
    if workspace !== nothing
        Tw = eltype(workspace)
        M = Contourlets._ws_matrix_type(workspace)
        image = Contourlets._allocate_zeros(M, Tw, workspace.image_size)
        return ct_inverse!(image, coeffs, params; workspace = workspace)
    end
    Tf = filter_eltype(T)
    p = Contourlets._convert_params(Tf, params)
    fp, qfp = p.lp_filters, p.dfb_filters
    J = p.J
    coarse = copy(coeffs.coarse)
    for j in J:-1:1
        bp = dfb_reconstruct(coeffs.subbands[j], qfp)
        coarse = lp_reconstruct(coarse, bp, fp)
    end
    return coarse
end

# ── Nonsubsampled Contourlet Transform ────────────────────────────────────────

"""
    nsct_forward(image::_AbstractGPUMatrix, params::ContourletParams) -> NSCTCoefficients

GPU Nonsubsampled Contourlet Transform forward pass (GPU pyramid + GPU NSDFB).
"""
function Contourlets.nsct_forward(
        image::_AbstractGPUMatrix, params::ContourletParams;
        workspace::Union{Contourlets.NSCTWorkspace, Nothing} = nothing,
        threaded::Union{Bool, Nothing} = nothing
    )
    if workspace !== nothing
        Tw = eltype(workspace)
        M = Contourlets._ws_matrix_type(workspace)
        coeffs = Contourlets.similar_nsct_coefficients(params, size(image); Td = Tw, M = M)
        img = Tw === eltype(image) ? image : Tw.(image)
        return nsct_forward!(coeffs, img, params; workspace = workspace, threaded = threaded)
    end
    Td = data_eltype(image)        # data type (real or complex)
    Tf = filter_eltype(Td)         # real filter precision
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
    return NSCTCoefficients(coarse, subbands)   # device-resident coeffs
end

# ── Moving whole coefficient sets between host and device ─────────────────────
# `Adapt.adapt(CuArray, coeffs)` moves the data arrays to the device and
# `Adapt.adapt(Array, coeffs)` brings them back.

function Adapt.adapt_structure(to, c::ContourletCoefficients)
    return ContourletCoefficients(
        Adapt.adapt(to, c.coarse),
        [[Adapt.adapt(to, s) for s in lvl] for lvl in c.subbands],
    )
end

# NSCT and TICT subbands stored as one array (`_stacked_zeros`) stay so where
# the target array type allows; otherwise each subband moves on its own.
function Adapt.adapt_structure(to, c::NSCTCoefficients)
    coarse = Adapt.adapt(to, c.coarse)
    storage, flat = _adapt_subbands(to, c.storage, coarse, reduce(vcat, c.subbands))
    return NSCTCoefficients(coarse, Contourlets._split(flat, length.(c.subbands)), storage)
end

function Adapt.adapt_structure(to, c::TICTCoefficients)
    coarse = Adapt.adapt(to, c.coarse)
    channels = reduce(vcat, c.subbands)
    storage, flat = _adapt_subbands(to, c.storage, coarse, reduce(vcat, channels))
    subbands = Contourlets._split(Contourlets._split(flat, length.(channels)), length.(c.subbands))
    return TICTCoefficients(coarse, subbands, storage)
end

# The subbands `subs`, moved as one array when a slice of an array like the moved
# `coarse` has its type: their storage `P` while they are still its slices, or
# host subbands of one size, stacked for the move.  Storage already of that type
# stays as it is.
function _adapt_subbands(to, P, coarse, subs::Vector)
    A = typeof(coarse)
    stacked = P !== nothing && all(is_last_axis_slice(P, s, i) for (i, s) in enumerate(subs))
    stacked && all(s -> s isa A, subs) && return P, A[s for s in subs]
    if !(coarse isa Array) && eltype(coarse) == eltype(first(subs)) &&
            (stacked || all(s -> s isa Array && size(s) == size(first(subs)), subs))
        Q = similar(coarse, (size(first(subs))..., length(subs)))
        slices = Contourlets._slices_as(Q, A)
        if slices !== nothing
            copyto!(Q, stacked ? P : stack(subs))
            return Q, slices
        end
    end
    return nothing, A[Adapt.adapt(to, s) for s in subs]
end
