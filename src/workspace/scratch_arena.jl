# Scratch arena: a grow-once buffer pool, the task-local active arena and filter
# cache, and the `_scratch_like` / `_allocate_zeros` helpers that draw from them.

# ── Scratch arena ─────────────────────────────────────────────────────────────
# A grow-once pool of `Matrix{T}` scratch buffers for the directional stage,
# whose transient buffers vary in number/size per node.  The allocation sequence
# is deterministic for fixed params + image size, so `reset!` at the start of a
# transform and `acquire!` hand back the same buffers on every subsequent call —
# zero allocations after the first.  Buffers are handed out by a monotonic cursor
# (never released mid-transform), so all live intermediates are distinct.

# Parametrised on the stored matrix type `M` so the same pool serves host
# (`Matrix`) and device (`CuMatrix`, …) buffers; new buffers are made with
# `similar(ref, …)` from a caller-supplied reference array.
mutable struct ScratchArena{M <: AbstractMatrix}
    bufs::Vector{M}
    cursor::Int
end
ScratchArena{M}() where {M <: AbstractMatrix} = ScratchArena{M}(M[], 0)

_arena_reset!(a::ScratchArena) = (a.cursor = 0; return a)

# Hand out a scratch `m×n` matrix shaped like `ref`, growing/resizing as needed.
function _arena_take!(a::ScratchArena{M}, ref, m::Int, n::Int) where {M}
    a.cursor += 1
    if a.cursor > length(a.bufs)
        push!(a.bufs, similar(ref, m, n)::M)
    elseif size(a.bufs[a.cursor]) != (m, n)
        a.bufs[a.cursor] = similar(ref, m, n)::M
    end
    return a.bufs[a.cursor]
end

# Task-local "active" arena.  Rather than thread an arena argument through every
# call, the pyramid stage of the workspace paths (and the device separable
# convolution's scratch) draws transient buffers from whatever arena is active on
# the current task, via `_scratch_like`.  When no matching arena is active
# `_scratch_like` falls back to `similar` (so the non-workspace paths are
# unchanged).  The arena is typed on its matrix type, so a host arena only serves
# host arrays.  Task-local ⇒ thread-safe.

_active_arena() = get(task_local_storage(), :_contourlets_scratch, nothing)

_active_filter_cache() = get(task_local_storage(), :_contourlets_filter_cache, nothing)

function _with_filter_cache(f, cache)
    store = task_local_storage()
    prev = get(store, :_contourlets_filter_cache, nothing)
    store[:_contourlets_filter_cache] = cache
    try
        return f()
    finally
        store[:_contourlets_filter_cache] = prev
    end
end

function _with_arena(f, arena)
    store = task_local_storage()
    prev = get(store, :_contourlets_scratch, nothing)
    store[:_contourlets_scratch] = arena
    try
        return f()
    finally
        store[:_contourlets_scratch] = prev
    end
end

# A scratch `m×n` matrix shaped like `ref`: drawn from the active arena when its
# stored matrix type matches `typeof(ref)`, otherwise `similar(ref, m, n)`.  The
# type match means a host arena only ever serves host arrays and a device arena
# only device arrays, so mixed/absent arenas fall back to `similar` safely.
_base_array_type(::Type{T}) where {T} = T
_base_array_type(::Type{SubArray{T, N, P, I, L}}) where {T, N, P, I, L} = _base_array_type(P)

function _scratch_like(ref::R, m::Int, n::Int) where {R <: AbstractMatrix}
    a = _active_arena()
    B = _base_array_type(R)
    if a isa ScratchArena{B}
        return _arena_take!(a, ref, m, n)
    end
    return similar(ref, m, n)
end

function _allocate_zeros(::Type{M}, Td::Type, dims::Tuple) where {M <: AbstractMatrix}
    T_concrete = M isa UnionAll ? M{Td} : M
    if T_concrete <: Array
        return zeros(Td, dims...)
    end
    dummy = similar(T_concrete, (1, 1))
    arr = similar(dummy, Td, dims)
    return fill!(arr, zero(Td))
end
