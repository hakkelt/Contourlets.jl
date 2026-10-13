# Nonsubsampled Pyramid (NSP) — à trous upsampled filters, no downsampling.
#
# All filtering uses *periodic* boundary extension.  Together with the periodic
# NSDFB this makes the NSCT exactly invariant under circular shifts of the
# input (the defining property of the nonsubsampled transform).
#
# The a trous filter pair per pyramid level, and the synthesis rescale it carries,
# come from MultiscaleFilterBanks (`nsp_atrous_filters`): the
# nonsubsampled pyramid does no decimation, so the biorthogonal synthesis `g` --
# designed with DC gain 2 to compensate the x2 upsampling of the decimated
# Laplacian pyramid -- is rescaled to unit DC gain per axis.  Without that the
# "bandpass" would be dominated by low-pass leakage instead of being a true
# bandpass, defeating the directional decomposition that follows.
#
# This follows the NSP of da Cunha, Zhou & Do (IEEE TIP 15(10), 2006, §II-A) with
# separable filters: the paper designs non-separable 2-D pyramid filters, this
# package uses the CDF 9/7 pair per axis.  À trous filtering: Shensa, IEEE TSP
# 40(10), 1992.

"""
    _nsp_masks(Tf, fp, dims, J; rows=1:dims[1]) -> (ana, syn)

The `J`-level NSP as masks on the DFT grid of `dims`, restricted to the frequency
rows `rows` (the `rfft` half-plane for real data).  Periodic filtering with the à
trous filters of level j is the pointwise product with their DTFTs Hⱼ, Gⱼ on the
grid, so the cascade folds into one mask per output: `ana[j]` takes the input
spectrum to the bandpass of level j,

    Bⱼ = (Π_{i<j} Hᵢ) (1 − GⱼHⱼ),        ana[J + 1] = Π_{i≤J} Hᵢ (the lowpass),

and `syn[j] = Π_{i<j} Gᵢ` (`syn[J + 1] = Π_{i≤J} Gᵢ`) takes it back, so that
`Σⱼ syn[j]·ana[j] = 1` telescopes.  This is [`nsp_decompose`](@ref) /
[`nsp_reconstruct`](@ref) applied level by level, to rounding error, at the cost
of one pointwise product per output instead of two convolutions per level.
"""
function _nsp_masks(
        ::Type{Tf}, fp::FilterPair, dims::NTuple{2, Int}, J::Int; rows = 1:dims[1]
    ) where {Tf <: AbstractFloat}
    a1, gh1, s1 = nsp_responses(fp, J, dims[1]; bins = rows)
    a2, gh2, s2 = nsp_responses(fp, J, dims[2])
    sep(u, v) = Matrix{Complex{Tf}}(u .* transpose(v))
    ana = [Matrix{Complex{Tf}}(a1[j] .* transpose(a2[j]) .* (1 .- gh1[j] .* transpose(gh2[j]))) for j in 1:J]
    push!(ana, sep(a1[J + 1], a2[J + 1]))
    syn = [sep(s1[j], s2[j]) for j in 1:(J + 1)]
    return ana, syn
end

"""
    nsp_decompose(image, fp::FilterPair, level::Int) -> (coarse, bandpass)

One level of the Nonsubsampled Pyramid at pyramid level `level` (1 = finest).
Both outputs have the same spatial size as `image`.

# Examples
```jldoctest
julia> using Contourlets, Random

julia> x = randn(Xoshiro(1), 32, 32);

julia> c, bp = nsp_decompose(x, CDF97, 1);

julia> size(c) == size(bp) == size(x)
true
```
"""
function nsp_decompose(image::AbstractMatrix, fp::FilterPair, level::Int)
    level >= 1 || throw(ArgumentError("level must be ≥ 1"))
    Td = data_eltype(image)   # data type (preserves complex)
    Tf = filter_eltype(Td)    # real filter precision
    im = Td === eltype(image) ? image : Td.(image)
    h_j, g_j = nsp_atrous_filters(fp, level, Tf)
    coarse = conv2d_sep(im, h_j, h_j; boundary = :periodic)
    pred = conv2d_sep(coarse, g_j, g_j; boundary = :periodic)
    bp = im .- pred
    return coarse, bp
end

"""
    nsp_decompose!(coarse, bandpass, image, fp::FilterPair, level;
                   tmp=similar(image), tmp2=similar(image)) -> (coarse, bandpass)

In-place NSP decomposition.  `tmp` and `tmp2` must be distinct buffers of the
same size as `image`; when both are supplied the call is allocation-free
(except for the à trous filter vectors).
"""
function nsp_decompose!(
        coarse::AbstractMatrix, bandpass::AbstractMatrix,
        image::AbstractMatrix, fp::FilterPair, level::Int;
        tmp::AbstractMatrix = similar(image),
        tmp2::AbstractMatrix = similar(image),
        h_j::Union{AbstractVector, Nothing} = nothing,
        g_j::Union{AbstractVector, Nothing} = nothing
    )
    # Use caller-supplied (workspace-cached) à trous filters when given; otherwise
    # build them here.  The cached `g_j` already carries the synthesis rescale.
    if h_j === nothing || g_j === nothing
        h_built, g_built = nsp_atrous_filters(fp, level, filter_eltype(eltype(coarse)))
        h_j = h_j === nothing ? h_built : h_j
        g_j = g_j === nothing ? g_built : g_j
    end
    conv2d_sep!(coarse, image, h_j, h_j; boundary = :periodic, tmp = tmp2)
    conv2d_sep!(tmp, coarse, g_j, g_j; boundary = :periodic, tmp = tmp2)
    @. bandpass = image - tmp
    return coarse, bandpass
end

"""
    nsp_reconstruct(coarse, bandpass, fp::FilterPair, level) -> image

Reconstruct one NSP level.

# Examples
```jldoctest
julia> using Contourlets, Random

julia> x = randn(Xoshiro(1), 32, 32);

julia> c, bp = nsp_decompose(x, CDF97, 1);

julia> rec = nsp_reconstruct(c, bp, CDF97, 1);

julia> maximum(abs, rec .- x) < 1e-12
true
```
"""
function nsp_reconstruct(
        coarse::AbstractMatrix, bandpass::AbstractMatrix,
        fp::FilterPair, level::Int
    )
    Td = promote_type(eltype(coarse), eltype(bandpass))  # data type
    Tf = filter_eltype(Td)                              # real filter precision
    coarse_T = Td === eltype(coarse) ? coarse : Td.(coarse)
    bandpass_T = Td === eltype(bandpass) ? bandpass : Td.(bandpass)
    _, g_j = nsp_atrous_filters(fp, level, Tf)
    pred = conv2d_sep(coarse_T, g_j, g_j; boundary = :periodic)
    return bandpass_T .+ pred
end

"""
    nsp_reconstruct!(image, coarse, bandpass, fp::FilterPair, level;
                     tmp=similar(image), tmp2=similar(image)) -> image

In-place NSP reconstruction.  See [`nsp_decompose!`](@ref) for the buffer
requirements.
"""
function nsp_reconstruct!(
        image::AbstractMatrix, coarse::AbstractMatrix,
        bandpass::AbstractMatrix, fp::FilterPair, level::Int;
        tmp::AbstractMatrix = similar(image),
        tmp2::AbstractMatrix = similar(image),
        g_j::Union{AbstractVector, Nothing} = nothing
    )
    # Use the caller-supplied (workspace-cached) synthesis filter when given.
    if g_j === nothing
        _, g_j = nsp_atrous_filters(fp, level, filter_eltype(eltype(image)))
    end
    conv2d_sep!(tmp, coarse, g_j, g_j; boundary = :periodic, tmp = tmp2)
    @. image = bandpass + tmp
    return image
end
