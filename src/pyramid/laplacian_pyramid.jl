# Laplacian Pyramid decomposition and reconstruction.
#
# Analysis (one level):
#   1. coarse  = rect_downsample( conv2d_sep(image, h, h) )
#   2. prediction = conv2d_sep( rect_upsample(coarse), g, g )
#   3. bandpass = image − prediction
#
# Reconstruction is always exact regardless of filter choice:
#   rec = bandpass + prediction = (image − prediction) + prediction = image

"""
    lp_decompose(image, fp::FilterPair) -> (coarse, bandpass)

One-level Laplacian Pyramid analysis.

# Examples
```jldoctest
julia> using Contourlets, Random; Random.seed!(42)
julia> x = randn(64, 64)
julia> c, bp = lp_decompose(x, CDF97)
julia> size(c)
(32, 32)
julia> size(bp) == size(x)
true
```
"""
function lp_decompose(image::AbstractMatrix, fp::FilterPair)
    T = float(eltype(image))   # preserve image precision (Float32 → Float32)
    im = T.(image)
    h = T.(fp.h); g = T.(fp.g)
    coarse = rect_downsample(conv2d_sep(im, h, h))
    pred = conv2d_sep(rect_upsample(coarse), g, g)
    n1, n2 = size(im)
    bandpass = similar(im)
    @inbounds for j in 1:n2, i in 1:n1
        bandpass[i, j] = im[i, j] - pred[i, j]
    end
    return coarse, bandpass
end

"""
    lp_decompose!(coarse, bandpass, image, fp::FilterPair; tmp) -> (coarse, bandpass)

In-place Laplacian Pyramid analysis. `tmp` must be `similar(image)`.
"""
function lp_decompose!(
        coarse::AbstractMatrix, bandpass::AbstractMatrix,
        image::AbstractMatrix, fp::FilterPair;
        tmp::AbstractMatrix = similar(image)
    )
    h = eltype(image).(fp.h); g = eltype(image).(fp.g)
    conv2d_sep!(tmp, image, h, h)
    rect_downsample!(coarse, tmp)
    up = rect_upsample(coarse)
    conv2d_sep!(tmp, up, g, g)
    n1, n2 = size(image)
    @inbounds for j in 1:n2, i in 1:n1
        bandpass[i, j] = image[i, j] - tmp[i, j]
    end
    return coarse, bandpass
end

"""
    lp_reconstruct(coarse, bandpass, fp::FilterPair) -> image

One-level Laplacian Pyramid synthesis.

# Examples
```jldoctest
julia> using Contourlets, Random; Random.seed!(42)
julia> x = randn(64, 64)
julia> c, bp = lp_decompose(x, CDF97)
julia> rec = lp_reconstruct(c, bp, CDF97)
julia> maximum(abs, rec .- x) < 1e-12
true
```
"""
function lp_reconstruct(
        coarse::AbstractMatrix, bandpass::AbstractMatrix,
        fp::FilterPair
    )
    T = promote_type(eltype(coarse), eltype(bandpass))  # preserve image precision
    g = T.(fp.g)
    pred = conv2d_sep(rect_upsample(coarse), g, g)
    n1, n2 = size(bandpass)
    out = similar(bandpass, T)
    @inbounds for j in 1:n2, i in 1:n1
        out[i, j] = bandpass[i, j] + pred[i, j]
    end
    return out
end

"""
    lp_reconstruct!(image, coarse, bandpass, fp::FilterPair; tmp) -> image

In-place LP reconstruction.
"""
function lp_reconstruct!(
        image::AbstractMatrix, coarse::AbstractMatrix,
        bandpass::AbstractMatrix, fp::FilterPair;
        tmp::AbstractMatrix = similar(image)
    )
    g = eltype(image).(fp.g)
    n1, n2 = size(bandpass)
    up = rect_upsample(coarse)
    conv2d_sep!(image, up, g, g)
    @inbounds for j in 1:n2, i in 1:n1
        image[i, j] = bandpass[i, j] + image[i, j]
    end
    return image
end
