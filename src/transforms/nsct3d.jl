# 3-D Nonsubsampled Contourlet Transform.
#
# Multiscale stage: `nsp3d_decompose` (à trous, no downsampling).
# Directional stage: `nsdfb3d_decompose` (hourglass + wedge refinement).
#
# Every coefficient array keeps the spatial size of the input, so the transform
# is exactly equivariant under circular shifts of the volume — the same design
# decision as the 2-D NSCT, and the reason both stages filter periodically.
#
# The directional bank depends only on the volume size, never on the scale, so
# the (large) hourglass mapping response is planned once and shared across every
# scale; only the wedge masks — matrices — differ per scale.

"""
    NSCT3DWorkspace{Td, T}

Reusable plans and buffers for the 3-D NSCT at one volume size and element type.
Build with [`make_nsct3d_workspace`](@ref) and pass it to `nsct3d_forward` /
`nsct3d_inverse` to keep repeated calls out of the allocator — the pattern
iterative reconstruction algorithms need.
"""
struct NSCT3DWorkspace{
        Td <: Number, T <: AbstractFloat, B <: NSDFB3D{T}, P,
        A3 <: AbstractArray{Td, 3},
    }
    dims::NTuple{3, Int}
    banks::Vector{B}                 # one per scale, sharing one hourglass bank
    conv::P                          # Conv3DFFTPlan for the pyramid stage
    input::A3                        # running input of the pyramid cascade
    coarse::A3
    bandpass::A3
    tmp::A3
    tmp2::A3
end

Base.eltype(::NSCT3DWorkspace{Td}) where {Td} = Td

"""
    make_nsct3d_workspace(params::ContourletParams3D, dims::NTuple{3,Int},
                         ::Type{Td}=Float64; threading=Auto(), flags=FFTW.MEASURE)

Allocate the workspace for `params` at volume size `dims` and element type `Td`
(real or complex).

The hourglass mapping response dominates the memory: one `real(Td)` array on the
rfft grid, `(n₁÷2+1) × n₂ × n₃`.  Everything per-scale is matrices.
"""
function make_nsct3d_workspace(
        params::ContourletParams3D, dims::NTuple{3, Int}, ::Type{Td} = Float64;
        threading::ThreadingPolicy = Auto(),
        flags::Integer = FFTW.MEASURE
    ) where {Td <: Number}
    T = _filter_eltype(Td)
    fb = make_hourglass3d(
        dims, T; order = params.order, poly = params.poly, tight = params.tight,
        threading = threading, flags = flags
    )
    banks = [
        nsdfb3d_sharing(fb, params.L_array[j]; poly = params.poly, transition = params.transition)
            for j in 1:params.J
    ]
    conv = make_conv3d_fft_plan(dims, Td; threading = threading, flags = flags)
    bufs = ntuple(_ -> Array{Td, 3}(undef, dims), 5)
    return NSCT3DWorkspace{Td, T, eltype(banks), typeof(conv), typeof(bufs[1])}(
        dims, banks, conv, bufs[1], bufs[2], bufs[3], bufs[4], bufs[5]
    )
end

make_nsct3d_workspace(params::ContourletParams3D, dims::Vararg{Int, 3}; kwargs...) =
    make_nsct3d_workspace(params, dims; kwargs...)

# Build a workspace on demand when the caller did not supply one.
function _nsct3d_workspace(
        ws::Union{NSCT3DWorkspace, Nothing}, params::ContourletParams3D,
        dims::NTuple{3, Int}, ::Type{Td}
    ) where {Td}
    ws === nothing && return make_nsct3d_workspace(params, dims, Td)
    ws.dims == dims || throw(DimensionMismatch("workspace size does not match the volume"))
    eltype(ws) === Td || throw(ArgumentError("workspace element type does not match the volume"))
    length(ws.banks) == params.J ||
        throw(ArgumentError("workspace was built for a different number of scales"))
    # `tight` is baked into the banks at build time and read back off them by
    # every mask, so a workspace built the other way silently normalises the
    # coefficients differently from what `params` asks for.
    ws.banks[1].fb.tight == params.tight || throw(
        ArgumentError(
            "workspace was built with tight = $(ws.banks[1].fb.tight), not $(params.tight)"
        )
    )
    # The per-scale directional depth has to match too: the banks decide how many
    # subbands are produced while `params` decides how many the caller allocates,
    # so a mismatch would either overrun the coefficient vector or leave part of
    # it undefined.
    for j in 1:params.J
        sum(ws.banks[j].levels) == params.L_array[j] || throw(
            ArgumentError(
                "workspace scale $(j) was built for L = $(sum(ws.banks[j].levels)), not $(params.L_array[j])"
            )
        )
    end
    return ws
end

"""
    nsct3d_forward(volume, params::ContourletParams3D; workspace=nothing) -> NSCT3DCoefficients

3-D Nonsubsampled Contourlet Transform.  Invariant under circular shifts of the
input; every subband keeps the size of `volume`.

See [`nsct3d_forward(f, volume, params)`](@ref) for the streaming form, which
does not hold all subbands at once.

# Examples
```jldoctest
julia> using Contourlets, Random

julia> x = randn(Xoshiro(5), 16, 16, 16);

julia> p = ContourletParams3D(J = 2, L_array = [1, 2], order = 6);

julia> c = nsct3d_forward(x, p);

julia> length(c.subbands[1]), length(c.subbands[2])
(6, 12)

julia> maximum(abs, nsct3d_inverse(c, p) .- x) < 1e-10
true
```
"""
function nsct3d_forward(
        volume::AbstractArray{<:Number, 3}, params::ContourletParams3D;
        workspace::Union{NSCT3DWorkspace, Nothing} = nothing
    )
    # Pin the container element type to the *data* type the transform works in,
    # not `eltype(volume)`: an integer volume is filtered as `float(eltype)`, and
    # storing those coefficients back into an integer array would throw.
    A = typeof(similar(volume, _data_eltype(volume)))
    subbands = [Vector{A}(undef, nsct3d_nsubbands(params, j)) for j in 1:params.J]
    coarse = nsct3d_forward(volume, params; workspace = workspace) do j, k, band
        subbands[j][k] = copy(band)
    end
    return NSCT3DCoefficients(copy(coarse), subbands)
end

"""
    nsct3d_forward(f, volume, params::ContourletParams3D; workspace=nothing)

Streaming 3-D NSCT: calls `f(j, k, subband)` for directional subband `k` of
scale `j`, reusing one buffer between calls.  `subband` is only valid until `f`
returns — copy or reduce it there.

This is the form to use at working sizes.  At 256³ `Float32` with `J = 3` and
`L = [2, 2, 3]` the full coefficient set is 3.1 GB, while streaming holds two
volumes plus the plans.

Returns the coarse residual.

# Examples
```jldoctest
julia> using Contourlets, Random

julia> x = randn(Xoshiro(6), 16, 16, 16);

julia> p = ContourletParams3D(J = 2, L_array = [1, 1], order = 6);

julia> total = 0.0;

julia> coarse = nsct3d_forward(x, p) do j, k, band
           global total += sum(abs2, band)
       end;

julia> total > 0 && size(coarse) == size(x)
true
```
"""
function nsct3d_forward(
        f, volume::AbstractArray{<:Number, 3}, params::ContourletParams3D;
        workspace::Union{NSCT3DWorkspace, Nothing} = nothing
    )
    Td = _data_eltype(volume)
    dims = size(volume)
    ws = _nsct3d_workspace(workspace, params, dims, Td)
    p = _convert_params3d(_filter_eltype(Td), params)

    copyto!(ws.input, volume)
    for j in 1:params.J
        h_j, g_j = _nsp3d_atrous_filters(p.lp_filters, j, _filter_eltype(Td))
        nsp3d_decompose!(
            ws.coarse, ws.bandpass, ws.input, p.lp_filters, j;
            tmp = ws.tmp, tmp2 = ws.tmp2, h_j = h_j, g_j = g_j, plan = ws.conv
        )
        nsdfb3d_decompose(ws.bandpass, ws.banks[j]) do k, band
            f(j, k, band)
        end
        copyto!(ws.input, ws.coarse)
    end
    return ws.coarse
end

"""
    nsct3d_inverse(coeffs::NSCT3DCoefficients, params::ContourletParams3D; workspace=nothing) -> volume

Invert [`nsct3d_forward`](@ref).
"""
function nsct3d_inverse(
        coeffs::NSCT3DCoefficients, params::ContourletParams3D;
        workspace::Union{NSCT3DWorkspace, Nothing} = nothing
    )
    out = similar(coeffs.coarse)
    return nsct3d_inverse!(out, coeffs, params; workspace = workspace)
end

"""
    nsct3d_inverse!(volume, coeffs, params::ContourletParams3D; workspace=nothing) -> volume

In-place [`nsct3d_inverse`](@ref).
"""
function nsct3d_inverse!(
        volume::AbstractArray{<:Number, 3}, coeffs::NSCT3DCoefficients,
        params::ContourletParams3D; workspace::Union{NSCT3DWorkspace, Nothing} = nothing
    )
    return nsct3d_inverse!(volume, coeffs.coarse, params; workspace = workspace) do j, k
        return coeffs.subbands[j][k]
    end
end

"""
    nsct3d_inverse(f, coarse, params::ContourletParams3D; workspace=nothing) -> volume

Streaming inverse: `f(j, k)` must return directional subband `k` of scale `j`.
The subbands are consumed one at a time, in reverse scale order, so a caller
holding them on disk or regenerating them never materialises the full set.
"""
function nsct3d_inverse(
        f, coarse::AbstractArray{<:Number, 3}, params::ContourletParams3D;
        workspace::Union{NSCT3DWorkspace, Nothing} = nothing
    )
    out = similar(coarse)
    return nsct3d_inverse!(f, out, coarse, params; workspace = workspace)
end

"""
    nsct3d_inverse!(f, volume, coarse, params::ContourletParams3D; workspace=nothing) -> volume

In-place streaming inverse; see [`nsct3d_inverse(f, coarse, params)`](@ref).
"""
function nsct3d_inverse!(
        f, volume::AbstractArray{<:Number, 3}, coarse::AbstractArray{<:Number, 3},
        params::ContourletParams3D; workspace::Union{NSCT3DWorkspace, Nothing} = nothing
    )
    Td = _data_eltype(volume)
    dims = size(volume)
    size(coarse) == dims || throw(DimensionMismatch("coarse size does not match the output"))
    ws = _nsct3d_workspace(workspace, params, dims, Td)
    p = _convert_params3d(_filter_eltype(Td), params)

    copyto!(ws.coarse, coarse)
    for j in params.J:-1:1
        # Invert the directional stage back into the scale-j bandpass.  A bare
        # sum when the bank is not tight; a masked accumulation when it is.
        nsdfb3d_reconstruct!(ws.bandpass, ws.banks[j]) do k
            band = f(j, k)
            size(band) == dims || throw(DimensionMismatch("subband ($(j), $(k)) has the wrong size"))
            band
        end
        _, g_j = _nsp3d_atrous_filters(p.lp_filters, j, _filter_eltype(Td))
        nsp3d_reconstruct!(
            ws.input, ws.coarse, ws.bandpass, p.lp_filters, j;
            tmp = ws.tmp, tmp2 = ws.tmp2, g_j = g_j, plan = ws.conv
        )
        copyto!(ws.coarse, ws.input)
    end
    copyto!(volume, ws.coarse)
    return volume
end
