# Nonsubsampled Pyramid (NSP) — à trous upsampled filters, no downsampling.

"""
    nsp_decompose(image, fp::FilterPair, level::Int) -> (coarse, bandpass)

One level of the Nonsubsampled Pyramid at pyramid level `level` (1 = coarsest).
Both outputs have the same spatial size as `image`.

# Examples
```jldoctest
julia> using Contourlets, Random; Random.seed!(1)
julia> x = randn(32, 32)
julia> c, bp = nsp_decompose(x, CDF97, 1)
julia> size(c) == size(x)
true
```
"""
function nsp_decompose(image::AbstractMatrix, fp::FilterPair, level::Int)
    level >= 1 || throw(ArgumentError("level must be ≥ 1"))
    T = promote_type(eltype(image), eltype(fp))
    im = T.(image)
    factor = 2^(level - 1)
    h_j = T.(upsample_filter(fp.h, factor))
    g_j = T.(upsample_filter(fp.g, factor))
    coarse = conv2d_sep(im, h_j, h_j)
    pred = conv2d_sep(coarse, g_j, g_j)
    n1, n2 = size(im)
    bp = similar(im)
    @inbounds for j in 1:n2, i in 1:n1
        bp[i, j] = im[i, j] - pred[i, j]
    end
    return coarse, bp
end

"""
    nsp_decompose!(coarse, bandpass, image, fp::FilterPair, level; tmp) -> (coarse, bandpass)

In-place NSP decomposition. `tmp` must be `similar(image)`.
"""
function nsp_decompose!(
        coarse::AbstractMatrix, bandpass::AbstractMatrix,
        image::AbstractMatrix, fp::FilterPair, level::Int;
        tmp::AbstractMatrix = similar(image)
    )
    factor = 2^(level - 1)
    T = eltype(coarse)
    h_j = T.(upsample_filter(fp.h, factor))
    g_j = T.(upsample_filter(fp.g, factor))
    conv2d_sep!(coarse, image, h_j, h_j)
    conv2d_sep!(tmp, coarse, g_j, g_j)
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
julia> using Contourlets, Random; Random.seed!(1)
julia> x = randn(32, 32)
julia> c, bp = nsp_decompose(x, CDF97, 1)
julia> rec = nsp_reconstruct(c, bp, CDF97, 1)
julia> maximum(abs, rec .- x) < 1e-12
true
```
"""
function nsp_reconstruct(
        coarse::AbstractMatrix, bandpass::AbstractMatrix,
        fp::FilterPair, level::Int
    )
    T = promote_type(eltype(coarse), eltype(bandpass), eltype(fp))
    factor = 2^(level - 1)
    g_j = T.(upsample_filter(fp.g, factor))
    pred = conv2d_sep(T.(coarse), g_j, g_j)
    n1, n2 = size(bandpass)
    out = similar(bandpass, T)
    @inbounds for j in 1:n2, i in 1:n1
        out[i, j] = bandpass[i, j] + pred[i, j]
    end
    return out
end

"""
    nsp_reconstruct!(image, coarse, bandpass, fp::FilterPair, level; tmp) -> image

In-place NSP reconstruction.
"""
function nsp_reconstruct!(
        image::AbstractMatrix, coarse::AbstractMatrix,
        bandpass::AbstractMatrix, fp::FilterPair, level::Int;
        tmp::AbstractMatrix = similar(image)
    )
    T = eltype(image)
    factor = 2^(level - 1)
    g_j = T.(upsample_filter(fp.g, factor))
    conv2d_sep!(tmp, coarse, g_j, g_j)
    n1, n2 = size(bandpass)
    @inbounds for j in 1:n2, i in 1:n1
        image[i, j] = bandpass[i, j] + tmp[i, j]
    end
    return image
end
