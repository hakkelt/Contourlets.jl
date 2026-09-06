# 3-D Nonsubsampled Contourlet Transform.
#
# Multiscale stage: `nsp3_decompose` (à trous, no downsampling).
# Directional stage: `nsdfb3_decompose` (hourglass + wedge refinement).
#
# Every coefficient array keeps the spatial size of the input, so the transform
# is exactly equivariant under circular shifts of the volume — the same design
# decision as the 2-D NSCT, and the reason both stages filter periodically.
#
# The directional bank depends only on the volume size, never on the scale, so
# the (large) hourglass mapping response is planned once and shared across every
# scale; only the wedge masks — matrices — differ per scale.

"""
    NSCT3Workspace{Td, T}

Reusable plans and buffers for the 3-D NSCT at one volume size and element type.
Build with [`make_nsct3_workspace`](@ref) and pass it to `nsct3_forward` /
`nsct3_inverse` to keep repeated calls out of the allocator — the pattern
iterative reconstruction algorithms need.
"""
struct NSCT3Workspace{Td <: Number, T <: AbstractFloat, FB, P}
    dims::NTuple{3, Int}
    banks::Vector{NSDFB3{T, FB}}     # one per scale, sharing one hourglass bank
    conv::P                          # Conv3DFFTPlan for the pyramid stage
    input::Array{Td, 3}              # running input of the pyramid cascade
    coarse::Array{Td, 3}
    bandpass::Array{Td, 3}
    tmp::Array{Td, 3}
    tmp2::Array{Td, 3}
end

Base.eltype(::NSCT3Workspace{Td}) where {Td} = Td

"""
    make_nsct3_workspace(params::ContourletParams3, dims::NTuple{3,Int},
                         ::Type{Td}=Float64; threading=Auto(), flags=FFTW.MEASURE)

Allocate the workspace for `params` at volume size `dims` and element type `Td`
(real or complex).

The hourglass mapping response dominates the memory: one `real(Td)` array on the
rfft grid, `(n₁÷2+1) × n₂ × n₃`.  Everything per-scale is matrices.
"""
function make_nsct3_workspace(
        params::ContourletParams3, dims::NTuple{3, Int}, ::Type{Td} = Float64;
        threading::ThreadingPolicy = Auto(),
        flags::Integer = FFTW.MEASURE
    ) where {Td <: Number}
    T = _filter_eltype(Td)
    fb = make_hourglass3(
        dims, T; order = params.order, poly = params.poly,
        threading = threading, flags = flags
    )
    banks = [
        nsdfb3_sharing(fb, params.L_array[j]; poly = params.poly, transition = params.transition)
            for j in 1:params.J
    ]
    conv = make_conv3d_fft_plan(dims, Td; threading = threading, flags = flags)
    bufs = ntuple(_ -> Array{Td, 3}(undef, dims), 5)
    return NSCT3Workspace{Td, T, typeof(fb), typeof(conv)}(
        dims, banks, conv, bufs[1], bufs[2], bufs[3], bufs[4], bufs[5]
    )
end

make_nsct3_workspace(params::ContourletParams3, dims::Vararg{Int, 3}; kwargs...) =
    make_nsct3_workspace(params, dims; kwargs...)

# Build a workspace on demand when the caller did not supply one.
function _nsct3_workspace(
        ws::Union{NSCT3Workspace, Nothing}, params::ContourletParams3,
        dims::NTuple{3, Int}, ::Type{Td}
    ) where {Td}
    ws === nothing && return make_nsct3_workspace(params, dims, Td)
    ws.dims == dims || throw(DimensionMismatch("workspace size does not match the volume"))
    eltype(ws) === Td || throw(ArgumentError("workspace element type does not match the volume"))
    length(ws.banks) == params.J ||
        throw(ArgumentError("workspace was built for a different number of scales"))
    return ws
end

"""
    nsct3_forward(volume, params::ContourletParams3; workspace=nothing) -> NSCT3Coefficients

3-D Nonsubsampled Contourlet Transform.  Invariant under circular shifts of the
input; every subband keeps the size of `volume`.

See [`nsct3_forward(f, volume, params)`](@ref) for the streaming form, which
does not hold all subbands at once.

# Examples
```jldoctest
julia> using Contourlets, Random

julia> x = randn(Xoshiro(5), 16, 16, 16);

julia> p = ContourletParams3(J = 2, L_array = [1, 2], order = 6);

julia> c = nsct3_forward(x, p);

julia> length(c.subbands[1]), length(c.subbands[2])
(6, 12)

julia> maximum(abs, nsct3_inverse(c, p) .- x) < 1e-10
true
```
"""
function nsct3_forward(
        volume::AbstractArray{<:Number, 3}, params::ContourletParams3;
        workspace::Union{NSCT3Workspace, Nothing} = nothing
    )
    Td = _data_eltype(volume)
    subbands = [Vector{Array{Td, 3}}(undef, nsct3_nsubbands(params, j)) for j in 1:params.J]
    coarse = nsct3_forward(volume, params; workspace = workspace) do j, k, band
        subbands[j][k] = copy(band)
    end
    return NSCT3Coefficients(copy(coarse), subbands)
end

"""
    nsct3_forward(f, volume, params::ContourletParams3; workspace=nothing)

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

julia> p = ContourletParams3(J = 2, L_array = [1, 1], order = 6);

julia> total = 0.0;

julia> coarse = nsct3_forward(x, p) do j, k, band
           global total += sum(abs2, band)
       end;

julia> total > 0 && size(coarse) == size(x)
true
```
"""
function nsct3_forward(
        f, volume::AbstractArray{<:Number, 3}, params::ContourletParams3;
        workspace::Union{NSCT3Workspace, Nothing} = nothing
    )
    Td = _data_eltype(volume)
    dims = size(volume)
    ws = _nsct3_workspace(workspace, params, dims, Td)
    p = _convert_params3(_filter_eltype(Td), params)

    copyto!(ws.input, volume)
    for j in 1:params.J
        h_j, g_j = _nsp3_atrous_filters(p.lp_filters, j, _filter_eltype(Td))
        nsp3_decompose!(
            ws.coarse, ws.bandpass, ws.input, p.lp_filters, j;
            tmp = ws.tmp, tmp2 = ws.tmp2, h_j = h_j, g_j = g_j, plan = ws.conv
        )
        nsdfb3_decompose(ws.bandpass, ws.banks[j]) do k, band
            f(j, k, band)
        end
        copyto!(ws.input, ws.coarse)
    end
    return ws.coarse
end

"""
    nsct3_inverse(coeffs::NSCT3Coefficients, params::ContourletParams3; workspace=nothing) -> volume

Invert [`nsct3_forward`](@ref).
"""
function nsct3_inverse(
        coeffs::NSCT3Coefficients, params::ContourletParams3;
        workspace::Union{NSCT3Workspace, Nothing} = nothing
    )
    out = similar(coeffs.coarse)
    return nsct3_inverse!(out, coeffs, params; workspace = workspace)
end

"""
    nsct3_inverse!(volume, coeffs, params::ContourletParams3; workspace=nothing) -> volume

In-place [`nsct3_inverse`](@ref).
"""
function nsct3_inverse!(
        volume::AbstractArray{<:Number, 3}, coeffs::NSCT3Coefficients,
        params::ContourletParams3; workspace::Union{NSCT3Workspace, Nothing} = nothing
    )
    return nsct3_inverse!(volume, coeffs.coarse, params; workspace = workspace) do j, k
        return coeffs.subbands[j][k]
    end
end

"""
    nsct3_inverse(f, coarse, params::ContourletParams3; workspace=nothing) -> volume

Streaming inverse: `f(j, k)` must return directional subband `k` of scale `j`.
The subbands are consumed one at a time, in reverse scale order, so a caller
holding them on disk or regenerating them never materialises the full set.
"""
function nsct3_inverse(
        f, coarse::AbstractArray{<:Number, 3}, params::ContourletParams3;
        workspace::Union{NSCT3Workspace, Nothing} = nothing
    )
    out = similar(coarse)
    return nsct3_inverse!(f, out, coarse, params; workspace = workspace)
end

"""
    nsct3_inverse!(f, volume, coarse, params::ContourletParams3; workspace=nothing) -> volume

In-place streaming inverse; see [`nsct3_inverse(f, coarse, params)`](@ref).
"""
function nsct3_inverse!(
        f, volume::AbstractArray{<:Number, 3}, coarse::AbstractArray{<:Number, 3},
        params::ContourletParams3; workspace::Union{NSCT3Workspace, Nothing} = nothing
    )
    Td = _data_eltype(volume)
    dims = size(volume)
    size(coarse) == dims || throw(DimensionMismatch("coarse size does not match the output"))
    ws = _nsct3_workspace(workspace, params, dims, Td)
    p = _convert_params3(_filter_eltype(Td), params)

    copyto!(ws.coarse, coarse)
    for j in params.J:-1:1
        # Sum the directional subbands back into the scale-j bandpass.
        fill!(ws.bandpass, zero(Td))
        for k in 1:nsct3_nsubbands(params, j)
            band = f(j, k)
            size(band) == dims || throw(DimensionMismatch("subband ($(j), $(k)) has the wrong size"))
            @. ws.bandpass += band
        end
        _, g_j = _nsp3_atrous_filters(p.lp_filters, j, _filter_eltype(Td))
        nsp3_reconstruct!(
            ws.input, ws.coarse, ws.bandpass, p.lp_filters, j;
            tmp = ws.tmp, tmp2 = ws.tmp2, g_j = g_j, plan = ws.conv
        )
        copyto!(ws.coarse, ws.input)
    end
    copyto!(volume, ws.coarse)
    return volume
end
