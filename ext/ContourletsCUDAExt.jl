module ContourletsCUDAExt

# CUDA graph capture for the workspace forward/inverse paths.
#
# The workspace paths of the CT and the STICT have a fixed kernel shape for a given
# (image size, params, eltype): same kernels, same sizes, same order every call.
# That is exactly the precondition for CUDA graph capture.  Both are many small
# FFTs and broadcasts (about 300 and 1,700 per call), where launches dominate.
#
# `_replay_or_capture!` keeps a few graphs per path (`_GraphCache`), each keyed on
# the device addresses of the input and of every output array: a new coefficient
# set may reuse some of the old addresses but not all.
#   - Hit: replay with a single `CUDA.launch(exec)`.
#   - Miss: run the path eagerly once, which writes a valid result and, for the
#     CT, warms its ScratchArena so the arena's device pointers are stable; then
#     capture and instantiate it, evicting the least recently used graph.
# A CT call that resizes an arena buffer (a call with other sizes) drops the
# cached graphs, which still point at the old buffer.  CPU-image inputs to the CT
# skip capture entirely, and so do the allocating `ct_forward(…; workspace)` and
# friends: their arrays are new on every call, so a graph would never replay.
#
# Nothing in the captured call may allocate: an allocation inside `CUDA.capture`
# becomes graph memory, which the pool later fails to free
# (`CUDA_ERROR_INVALID_VALUE` from `cuMemFreeAsync` at GC).  That is why the inverse
# FFTs are unnormalised `plan_bfft!` plans (a normalised plan scales through a
# device scalar) and why the CT captures with its arena reset (`_scratch_like`
# returns the warmed buffers) and with prebuilt device LP filters
# (`_with_filter_cache`; `_ensure_gpu` would allocate them).  The STICT's buffers
# and plans are all built with its workspace.
#
# (The volume transforms, now in Surfacelets.jl, are deliberately not captured:
# they are a few full-volume FFTs, and their streaming callbacks run host code.)

using Contourlets: Contourlets, CTWorkspace, ContourletCoefficients, STICTWorkspace, TICTCoefficients, _GraphCache
using CUDA: CUDA

# ── CT ────────────────────────────────────────────────────────────────────────

function Contourlets._ct_forward_ws!(
        coeffs::ContourletCoefficients{T},
        image::AbstractMatrix,
        ws::CTWorkspace{T, Tf, M}
    ) where {T, Tf <: AbstractFloat, M <: CUDA.CuMatrix{T}}
    # Only capture when the image is device-resident (same stream domain).
    if image isa CUDA.CuArray
        arena = ws.fwd_scratch
        _replay_or_capture!(ws.graph_cache, image, coeffs; prepare = f -> _ct_prepare(f, ws, arena), epoch = () -> arena.replaced) do
            Contourlets._ct_forward_inner!(coeffs, image, ws)
        end
    else
        Contourlets._ct_forward_inner!(coeffs, image, ws)
    end
    return coeffs
end

function Contourlets._ct_inverse_ws!(
        image::AbstractMatrix{T},
        coeffs::ContourletCoefficients{T},
        ws::CTWorkspace{T, Tf, M}
    ) where {T, Tf <: AbstractFloat, M <: CUDA.CuMatrix{T}}
    if image isa CUDA.CuArray
        arena = ws.inv_scratch
        _replay_or_capture!(ws.inv_graph_cache, image, coeffs; prepare = f -> _ct_prepare(f, ws, arena), epoch = () -> arena.replaced) do
            Contourlets._ct_inverse_inner!(image, coeffs, ws)
        end
    else
        Contourlets._ct_inverse_inner!(image, coeffs, ws)
    end
    return image
end

# Before a CT capture, after the eager pass: the device LP filters built outside
# the capture, so they are regular pool allocations, and the arena rewound, so the
# capture reuses its warmed buffers.  Returns `f` with the filters in reach, and
# the filters, which the graph reads.  `CuArray{Tf}`, not `cu`: `cu` makes
# Float64 filters Float32.
function _ct_prepare(f::F, ws::CTWorkspace{T, Tf}, arena) where {F, T, Tf}
    fp = ws.params.lp_filters
    filters = Base.IdDict{AbstractVector, Any}(fp.h => CUDA.CuArray{Tf}(fp.h), fp.g => CUDA.CuArray{Tf}(fp.g))
    Contourlets._arena_reset!(arena)
    return () -> Contourlets._with_filter_cache(f, filters), filters
end

# ── STICT ─────────────────────────────────────────────────────────────────────

function Contourlets._stict_forward_ws!(
        coeffs::TICTCoefficients, x::CUDA.CuMatrix, ws::STICTWorkspace{Td, Tf, C}
    ) where {Td <: Number, Tf <: AbstractFloat, C <: CUDA.CuMatrix{Complex{Tf}}}
    _replay_or_capture!(ws.graph_cache, x, coeffs) do
        Contourlets._stict_forward_inner!(coeffs, x, ws)
    end
    return coeffs
end

function Contourlets._stict_inverse_ws!(
        x::CUDA.CuMatrix, coeffs::TICTCoefficients, ws::STICTWorkspace{Td, Tf, C}
    ) where {Td <: Number, Tf <: AbstractFloat, C <: CUDA.CuMatrix{Complex{Tf}}}
    _replay_or_capture!(ws.inv_graph_cache, x, coeffs) do
        Contourlets._stict_inverse_inner!(x, coeffs, ws)
    end
    return x
end

# ── Replay or capture ─────────────────────────────────────────────────────────

# The subband arrays of a coefficient set.
_leaves(c::ContourletCoefficients) = Iterators.flatten(c.subbands)
_leaves(c::TICTCoefficients) = Iterators.flatten(Iterators.flatten(c.subbands))

# The device addresses of `x`, the lowpass and every subband, in order.
_pointers(x, c) = UInt[UInt(pointer(x)); UInt(pointer(c.coarse)); [UInt(pointer(s)) for s in _leaves(c)]]

# Whether `ptrs` are `_pointers(x, c)`, without allocating.
function _same_pointers(ptrs::Vector{UInt}, x, c)
    length(ptrs) >= 2 && ptrs[1] == UInt(pointer(x)) && ptrs[2] == UInt(pointer(c.coarse)) || return false
    n = 2
    for s in _leaves(c)
        n += 1
        n <= length(ptrs) && ptrs[n] == UInt(pointer(s)) || return false
    end
    return n == length(ptrs)
end

# Replays the graph of `cache` captured with the arrays of `x` and `c`; else runs
# `f` eagerly, then captures `g` from `(g, keep) = prepare(f)` and keeps `keep`
# (what the graph reads but no one else roots) with it.  `epoch()` is the state
# of the workspace buffers: graphs captured in another epoch are dropped.
function _replay_or_capture!(
        f::F, cache::_GraphCache, x, c; prepare::G = f -> (f, nothing), epoch::E = () -> 0
    ) where {F, G, E}
    entries = cache.entries
    if epoch() != cache.epoch
        empty!(entries)
    end
    for i in eachindex(entries)
        ptrs, exec, _ = entries[i]
        _same_pointers(ptrs, x, c) || continue
        i == 1 || pushfirst!(entries, popat!(entries, i))
        CUDA.launch(exec::CUDA.CuGraphExec)
        return nothing
    end
    f()
    # The eager pass may resize the workspace buffers: older graphs are stale.
    if epoch() != cache.epoch
        empty!(entries)
        cache.epoch = epoch()
    end
    g, keep = prepare(f)
    exec = CUDA.instantiate(CUDA.capture(g))
    # Only `keep` stays with the graph, not `g`, which holds the caller's arrays.
    pushfirst!(entries, (_pointers(x, c), exec, keep))
    length(entries) > Contourlets._GRAPH_CACHE_SIZE && pop!(entries)
    return nothing
end

end # module ContourletsCUDAExt
