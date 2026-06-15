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
    Td = _data_eltype(image)   # data type (preserves complex)
    Tf = _filter_eltype(Td)    # real filter precision
    # Convert only when the element type actually differs (T.(x) always copies).
    im = Td === eltype(image) ? image : Td.(image)
    h = Tf === eltype(fp) ? fp.h : Tf.(fp.h)
    g = Tf === eltype(fp) ? fp.g : Tf.(fp.g)
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
    Tf = _filter_eltype(eltype(image))
    h = Tf === eltype(fp) ? fp.h : Tf.(fp.h)
    g = Tf === eltype(fp) ? fp.g : Tf.(fp.g)
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
    Td = promote_type(eltype(coarse), eltype(bandpass))  # data type
    Tf = _filter_eltype(Td)                              # real filter precision
    g = Tf === eltype(fp) ? fp.g : Tf.(fp.g)
    coarse_T = Td === eltype(coarse) ? coarse : Td.(coarse)
    bandpass_T = Td === eltype(bandpass) ? bandpass : Td.(bandpass)
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
    Tf = _filter_eltype(eltype(image))
    g = Tf === eltype(fp) ? fp.g : Tf.(fp.g)
    n1, n2 = size(bandpass)
    rect_upsample!(tmp, coarse)
    conv2d_sep!(image, tmp, g, g; tmp = tmp2)
    @inbounds for j in 1:n2, i in 1:n1
        image[i, j] = bandpass[i, j] + image[i, j]
    end
    return image
end
