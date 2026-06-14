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
julia> using Contourlets, Random

julia> x = randn(Xoshiro(42), 64, 64);

julia> c, bp = lp_decompose(x, CDF97);

julia> size(c), size(bp) == size(x)
((32, 32), true)
```
"""
function lp_decompose(image::AbstractMatrix, fp::FilterPair)
    T = float(eltype(image))   # preserve image precision (Float32 → Float32)
    # Convert only when the element type actually differs (T.(x) always copies).
    im = T === eltype(image) ? image : T.(image)
    h = T === eltype(fp) ? fp.h : T.(fp.h)
    g = T === eltype(fp) ? fp.g : T.(fp.g)
    coarse = rect_downsample(conv2d_sep(im, h, h))
    pred = conv2d_sep(rect_upsample(coarse), g, g)
    bandpass = im .- pred
    return coarse, bandpass
end

"""
    lp_decompose!(coarse, bandpass, image, fp::FilterPair;
                  tmp=similar(image), tmp2=similar(image)) -> (coarse, bandpass)

In-place Laplacian Pyramid analysis.  `tmp` and `tmp2` must be distinct buffers
of the same size as `image` (which must have even dimensions); when both are
supplied the call is allocation-free.
"""
function lp_decompose!(
        coarse::AbstractMatrix, bandpass::AbstractMatrix,
        image::AbstractMatrix, fp::FilterPair;
        tmp::AbstractMatrix = similar(image),
        tmp2::AbstractMatrix = similar(image)
    )
    all(iseven, size(image)) ||
        throw(ArgumentError("lp_decompose! requires even image dimensions; use lp_decompose"))
    Ti = eltype(image)
    h = Ti === eltype(fp) ? fp.h : Ti.(fp.h)
    g = Ti === eltype(fp) ? fp.g : Ti.(fp.g)
    conv2d_sep!(tmp, image, h, h; tmp = tmp2)
    rect_downsample!(coarse, tmp)
    rect_upsample!(tmp, coarse)
    # `bandpass` doubles as the scratch buffer of the second separable pass;
    # it is fully overwritten by the difference loop below.
    conv2d_sep!(tmp2, tmp, g, g; tmp = bandpass)
    n1, n2 = size(image)
    @inbounds for j in 1:n2, i in 1:n1
        bandpass[i, j] = image[i, j] - tmp2[i, j]
    end
    return coarse, bandpass
end

"""
    lp_reconstruct(coarse, bandpass, fp::FilterPair) -> image

One-level Laplacian Pyramid synthesis.

# Examples
```jldoctest
julia> using Contourlets, Random

julia> x = randn(Xoshiro(42), 64, 64);

julia> c, bp = lp_decompose(x, CDF97);

julia> rec = lp_reconstruct(c, bp, CDF97);

julia> maximum(abs, rec .- x) < 1e-12
true
```
"""
function lp_reconstruct(
        coarse::AbstractMatrix, bandpass::AbstractMatrix,
        fp::FilterPair
    )
    T = promote_type(eltype(coarse), eltype(bandpass))  # preserve image precision
    g = T === eltype(fp) ? fp.g : T.(fp.g)
    coarse_T = T === eltype(coarse) ? coarse : T.(coarse)
    bandpass_T = T === eltype(bandpass) ? bandpass : T.(bandpass)
    pred = conv2d_sep(rect_upsample(coarse_T), g, g)
    return bandpass_T .+ pred
end

"""
    lp_reconstruct!(image, coarse, bandpass, fp::FilterPair;
                    tmp=similar(image), tmp2=similar(image)) -> image

In-place LP reconstruction.  See [`lp_decompose!`](@ref) for the buffer
requirements.
"""
function lp_reconstruct!(
        image::AbstractMatrix, coarse::AbstractMatrix,
        bandpass::AbstractMatrix, fp::FilterPair;
        tmp::AbstractMatrix = similar(image),
        tmp2::AbstractMatrix = similar(image)
    )
    Ti = eltype(image)
    g = Ti === eltype(fp) ? fp.g : Ti.(fp.g)
    n1, n2 = size(bandpass)
    rect_upsample!(tmp, coarse)
    conv2d_sep!(image, tmp, g, g; tmp = tmp2)
    @inbounds for j in 1:n2, i in 1:n1
        image[i, j] = bandpass[i, j] + image[i, j]
    end
    return image
end
