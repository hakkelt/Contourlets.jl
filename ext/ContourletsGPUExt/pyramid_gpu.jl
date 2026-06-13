# GPU overloads for the Laplacian Pyramid and Nonsubsampled Pyramid.
#
# These follow the same algorithm as the CPU versions but operate on GPU arrays.
# The key difference is that `rect_downsample`, `rect_upsample`, and `conv2d_sep`
# dispatch to GPU-specific kernels (defined in primitives_gpu.jl) when given
# GPU arrays.  Everything else (filter allocation, size computation) is unchanged.

# ── Laplacian Pyramid ────────────────────────────────────────────────────────

function lp_decompose(image::AbstractGPUMatrix, fp::FilterPair)
    T = promote_type(eltype(image), eltype(fp))
    backend = _gpu_backend(image)
    im = T.(image)
    h_d = _ensure_gpu(backend, T.(fp.h))
    g_d = _ensure_gpu(backend, T.(fp.g))
    coarse_full = similar(im)
    conv2d_sep!(coarse_full, im, h_d, h_d)
    coarse = rect_downsample(coarse_full)
    coarse_up = similar(im)
    coarse_full2 = similar(im)
    rect_upsample!(coarse_full2, coarse)
    conv2d_sep!(coarse_up, coarse_full2, g_d, g_d)
    bp = im .- coarse_up
    return coarse, bp
end

function lp_reconstruct(coarse::AbstractGPUMatrix, bandpass::AbstractGPUMatrix, fp::FilterPair)
    T = promote_type(eltype(coarse), eltype(bandpass), eltype(fp))
    backend = _gpu_backend(coarse)
    g_d = _ensure_gpu(backend, T.(fp.g))
    up = similar(bandpass, T)
    coarse_up = similar(bandpass, T)
    rect_upsample!(up, T.(coarse))
    conv2d_sep!(coarse_up, up, g_d, g_d)
    return T.(bandpass) .+ coarse_up
end

function lp_decompose!(
        coarse::AbstractGPUMatrix, bandpass::AbstractGPUMatrix,
        image::AbstractGPUMatrix, fp::FilterPair;
        tmp::AbstractGPUMatrix = similar(image)
    )
    T = eltype(coarse)
    backend = _gpu_backend(image)
    h_d = _ensure_gpu(backend, T.(fp.h))
    g_d = _ensure_gpu(backend, T.(fp.g))
    tmp_full = similar(image, T)
    conv2d_sep!(tmp_full, image, h_d, h_d)
    rect_downsample!(coarse, tmp_full)
    rect_upsample!(tmp, coarse)
    conv2d_sep!(tmp_full, tmp, g_d, g_d)
    bandpass .= image .- tmp_full
    return coarse, bandpass
end

function lp_reconstruct!(
        image::AbstractGPUMatrix, coarse::AbstractGPUMatrix,
        bandpass::AbstractGPUMatrix, fp::FilterPair;
        tmp::AbstractGPUMatrix = similar(image)
    )
    T = eltype(image)
    backend = _gpu_backend(coarse)
    g_d = _ensure_gpu(backend, T.(fp.g))
    up = similar(image, T)
    rect_upsample!(up, coarse)
    conv2d_sep!(tmp, up, g_d, g_d)
    image .= bandpass .+ tmp
    return image
end

# ── Nonsubsampled Pyramid ────────────────────────────────────────────────────

function nsp_decompose(image::AbstractGPUMatrix, fp::FilterPair, level::Int)
    level >= 1 || throw(ArgumentError("level must be ≥ 1"))
    T = promote_type(eltype(image), eltype(fp))
    backend = _gpu_backend(image)
    im = T.(image)
    factor = 2^(level - 1)
    h_j = T.(upsample_filter(fp.h, factor))
    g_j = T(Contourlets._NSP_SYNTH_SCALE) .* T.(upsample_filter(fp.g, factor))
    h_d = _ensure_gpu(backend, h_j)
    g_d = _ensure_gpu(backend, g_j)
    coarse = similar(im)
    conv2d_sep!(coarse, im, h_d, h_d; boundary = :periodic)
    pred = similar(im)
    conv2d_sep!(pred, coarse, g_d, g_d; boundary = :periodic)
    bp = im .- pred
    return coarse, bp
end

function nsp_reconstruct(
        coarse::AbstractGPUMatrix, bandpass::AbstractGPUMatrix,
        fp::FilterPair, level::Int
    )
    T = promote_type(eltype(coarse), eltype(bandpass), eltype(fp))
    backend = _gpu_backend(coarse)
    factor = 2^(level - 1)
    g_j = T(Contourlets._NSP_SYNTH_SCALE) .* T.(upsample_filter(fp.g, factor))
    g_d = _ensure_gpu(backend, g_j)
    pred = similar(bandpass, T)
    conv2d_sep!(pred, T.(coarse), g_d, g_d; boundary = :periodic)
    return T.(bandpass) .+ pred
end

function nsp_decompose!(
        coarse::AbstractGPUMatrix, bandpass::AbstractGPUMatrix,
        image::AbstractGPUMatrix, fp::FilterPair, level::Int;
        tmp::AbstractGPUMatrix = similar(image)
    )
    factor = 2^(level - 1)
    T = eltype(coarse)
    backend = _gpu_backend(image)
    h_j = _ensure_gpu(backend, T.(upsample_filter(fp.h, factor)))
    g_j = _ensure_gpu(backend, T(Contourlets._NSP_SYNTH_SCALE) .* T.(upsample_filter(fp.g, factor)))
    conv2d_sep!(coarse, image, h_j, h_j; boundary = :periodic)
    conv2d_sep!(tmp, coarse, g_j, g_j; boundary = :periodic)
    bandpass .= image .- tmp
    return coarse, bandpass
end

function nsp_reconstruct!(
        image::AbstractGPUMatrix, coarse::AbstractGPUMatrix,
        bandpass::AbstractGPUMatrix, fp::FilterPair, level::Int;
        tmp::AbstractGPUMatrix = similar(image)
    )
    T = eltype(image)
    backend = _gpu_backend(coarse)
    factor = 2^(level - 1)
    g_j = _ensure_gpu(backend, T(Contourlets._NSP_SYNTH_SCALE) .* T.(upsample_filter(fp.g, factor)))
    conv2d_sep!(tmp, coarse, g_j, g_j; boundary = :periodic)
    image .= bandpass .+ tmp
    return image
end
