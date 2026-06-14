# Nonsubsampled Pyramid (NSP) — à trous upsampled filters, no downsampling.
#
# All filtering uses *periodic* boundary extension.  Together with the periodic
# NSDFB this makes the NSCT exactly invariant under circular shifts of the
# input (the defining property of the nonsubsampled transform).
#
# Synthesis-filter normalisation: the analysis low-pass `h` has unit DC gain, but
# the biorthogonal synthesis `g` has DC gain 2 because it is designed to
# compensate the ×2 upsampling of the *decimated* Laplacian pyramid.  The
# nonsubsampled pyramid does no decimation, so `g` is rescaled to unit DC gain
# (`g/2` per axis).  Without this the "bandpass" would be dominated by low-pass
# leakage (prediction ≈ 2× the image on smooth regions) instead of being a true
# bandpass — defeating the directional decomposition that follows.
const _NSP_SYNTH_SCALE = 0.5

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
    T = promote_type(eltype(image), eltype(fp))
    im = T === eltype(image) ? image : T.(image)
    factor = 2^(level - 1)
    h_up = upsample_filter(fp.h, factor)
    h_j = T === eltype(h_up) ? h_up : T.(h_up)
    g_up = upsample_filter(fp.g, factor)
    g_j = T(_NSP_SYNTH_SCALE) .* (T === eltype(g_up) ? g_up : T.(g_up))
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
        tmp2::AbstractMatrix = similar(image)
    )
    factor = 2^(level - 1)
    T = eltype(coarse)
    # upsample_filter already returns a fresh vector; convert only on type mismatch.
    h_up = upsample_filter(fp.h, factor)
    h_j = T === eltype(h_up) ? h_up : T.(h_up)
    g_up = upsample_filter(fp.g, factor)
    g_j = T(_NSP_SYNTH_SCALE) .* (T === eltype(g_up) ? g_up : T.(g_up))
    conv2d_sep!(coarse, image, h_j, h_j; boundary = :periodic, tmp = tmp2)
    conv2d_sep!(tmp, coarse, g_j, g_j; boundary = :periodic, tmp = tmp2)
    n1, n2 = size(image)
    @inbounds for j in 1:n2, i in 1:n1
        bandpass[i, j] = image[i, j] - tmp[i, j]
    end
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
    T = promote_type(eltype(coarse), eltype(bandpass), eltype(fp))
    coarse_T = T === eltype(coarse) ? coarse : T.(coarse)
    bandpass_T = T === eltype(bandpass) ? bandpass : T.(bandpass)
    factor = 2^(level - 1)
    g_up = upsample_filter(fp.g, factor)
    g_j = T(_NSP_SYNTH_SCALE) .* (T === eltype(g_up) ? g_up : T.(g_up))
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
        tmp2::AbstractMatrix = similar(image)
    )
    T = eltype(image)
    factor = 2^(level - 1)
    g_up = upsample_filter(fp.g, factor)
    g_j = T(_NSP_SYNTH_SCALE) .* (T === eltype(g_up) ? g_up : T.(g_up))
    conv2d_sep!(tmp, coarse, g_j, g_j; boundary = :periodic, tmp = tmp2)
    n1, n2 = size(bandpass)
    @inbounds for j in 1:n2, i in 1:n1
        image[i, j] = bandpass[i, j] + tmp[i, j]
    end
    return image
end
