module ContourletsCUDAExt

# CUDA graph capture for ct_forward!.
#
# The workspace-bound `ct_forward!` has a fixed kernel shape for a given
# (image size, params, eltype): same kernels, same sizes, same order every call.
# That is exactly the precondition for CUDA graph capture.
#
# Strategy (first call on a given workspace):
#   1. Run `_ct_forward_inner!` eagerly once to warm the ScratchArena (populates
#      all arena slots so their device pointers are stable before capture) and
#      to produce a valid result in coeffs.
#   2. Pre-build device LP filter vectors (h_d, g_d) outside CUDA.capture() so
#      they are regular pool allocations, not graph memory nodes.
#   3. Reset the arena cursor and call `CUDA.capture(_ct_forward_inner!)` wrapped
#      in `_with_filter_cache` so conv2d_sep! finds the pre-built device vectors
#      instead of calling _ensure_gpu (which would create new pool allocations →
#      graph memory nodes → CUDA_ERROR_INVALID_VALUE on free).
#   4. Instantiate and cache [exec, img_ptr, coarse_ptr] in ws.graph_cache.
#
# Subsequent calls: if input/output device pointers match the cached ones,
# replay with a single `CUDA.launch(exec)`.  On mismatch (different pre-allocated
# buffers), re-capture.  CPU-image inputs skip graph capture entirely.

using Contourlets
using CUDA

import Contourlets:
    ct_forward!, ct_inverse!, ContourletWorkspace, ContourletCoefficients, ThreadingPolicy, Auto

function ct_forward!(
        coeffs::ContourletCoefficients{T},
        image::AbstractMatrix,
        ws::ContourletWorkspace{T, Tf, M};
        threading::ThreadingPolicy = Auto()
    ) where {T, Tf <: AbstractFloat, M <: CUDA.CuMatrix{T}}
    # Only capture when the image is device-resident (same stream domain).
    if !(image isa CUDA.CuArray)
        Contourlets._ct_forward_inner!(coeffs, image, ws; threading = threading)
        return coeffs
    end

    cache = ws.graph_cache
    img_ptr = UInt(pointer(image))
    coarse_ptr = UInt(pointer(coeffs.coarse))

    if length(cache) == 3 && cache[2]::UInt == img_ptr && cache[3]::UInt == coarse_ptr
        # Replay: single host launch replaces ~300 kernel launches.
        CUDA.launch(cache[1]::CUDA.CuGraphExec)
        return coeffs
    end

    # Miss: warm the arena then capture.
    # Pass 1 — eager execution; warms the ScratchArena and writes a valid result.
    Contourlets._ct_forward_inner!(coeffs, image, ws; threading = threading)

    # Pre-build device LP filter vectors OUTSIDE CUDA.capture().  These are
    # regular pool allocations; passing them via _with_filter_cache means
    # conv2d_sep! won't call _ensure_gpu during capture (which would create
    # graph-owned memory nodes that fail to free via the normal pool).
    fp = ws.params.lp_filters
    h_d = CUDA.cu(Tf.(fp.h))
    g_d = CUDA.cu(Tf.(fp.g))
    filter_cache = Base.IdDict{AbstractVector, Any}(fp.h => h_d, fp.g => g_d)

    # Pass 2 — arena reset + capture.  _scratch_like returns existing buffers
    # (no cuMemAllocAsync); filter cache provides stable device vectors.
    Contourlets._arena_reset!(ws.fwd_scratch)
    graph = CUDA.capture() do
        Contourlets._with_filter_cache(filter_cache) do
            Contourlets._ct_forward_inner!(coeffs, image, ws; threading = threading)
        end
    end
    exec = CUDA.instantiate(graph)

    empty!(cache)
    push!(cache, exec, img_ptr, coarse_ptr)
    return coeffs
end

function ct_inverse!(
        image::AbstractMatrix{T},
        coeffs::ContourletCoefficients{T},
        ws::ContourletWorkspace{T, Tf, M};
        threading::ThreadingPolicy = Auto()
    ) where {T, Tf <: AbstractFloat, M <: CUDA.CuMatrix{T}}
    if !(image isa CUDA.CuArray)
        Contourlets._ct_inverse_inner!(image, coeffs, ws; threading = threading)
        return image
    end

    cache = ws.inv_graph_cache
    image_ptr = UInt(pointer(image))
    coarse_ptr = UInt(pointer(coeffs.coarse))

    if length(cache) == 3 && cache[2]::UInt == image_ptr && cache[3]::UInt == coarse_ptr
        CUDA.launch(cache[1]::CUDA.CuGraphExec)
        return image
    end

    # Pass 1 — eager execution; warms the ScratchArena and writes a valid result.
    Contourlets._ct_inverse_inner!(image, coeffs, ws; threading = threading)

    fp = ws.params.lp_filters
    h_d = CUDA.cu(Tf.(fp.h))
    g_d = CUDA.cu(Tf.(fp.g))
    filter_cache = Base.IdDict{AbstractVector, Any}(fp.h => h_d, fp.g => g_d)

    # Pass 2 — arena reset + capture with stable filter pointers.
    Contourlets._arena_reset!(ws.inv_scratch)
    graph = CUDA.capture() do
        Contourlets._with_filter_cache(filter_cache) do
            Contourlets._ct_inverse_inner!(image, coeffs, ws; threading = threading)
        end
    end
    exec = CUDA.instantiate(graph)

    empty!(cache)
    push!(cache, exec, image_ptr, coarse_ptr)
    return image
end

end # module ContourletsCUDAExt
