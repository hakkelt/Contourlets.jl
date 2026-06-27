# CUDA Graph Capture (ContourletsCUDAExt)

## What It Does

The workspace-bound `_ct_forward_ws!` and `_ct_inverse_ws!` paths have a fixed
kernel shape for a given `(image_size, params, eltype)`. `ContourletsCUDAExt`
captures this sequence as a CUDA graph and replays it with a single host call,
replacing ~300 individual kernel launches.

## Capture Protocol (two-pass strategy)

**Pass 1 — eager warmup:**
- Run `_ct_forward_inner!` normally to populate all `ScratchArena` slots
  (arena buffers must be allocated before capture so their device pointers are stable).
- This also produces a valid result in `coeffs`.

**Pass 2 — capture:**
- Reset the arena cursor with `_arena_reset!(ws.fwd_scratch)`.
- Pre-build device LP filter vectors (`CUDA.cu(Tf.(fp.h))`) *outside* `CUDA.capture()`.
  Storing them in a `_with_filter_cache` call ensures `conv2d_sep!` finds the
  stable vectors instead of calling `_ensure_gpu` (which would create graph-owned
  memory nodes that fail to free via the normal pool).
- Call `CUDA.capture(_ct_forward_inner!)`.
- Instantiate and cache `[exec, img_ptr, coarse_ptr]` in `ws.graph_cache`.

**Replay (cache hit):**
- Check `img_ptr == pointer(image)` and `coarse_ptr == pointer(coeffs.coarse)`.
- If both match: `CUDA.launch(exec)` — single host call.
- On mismatch: re-capture (different pre-allocated buffers).

## Preconditions for Capture to Trigger

1. The workspace must be CUDA-resident (`M <: CUDA.CuMatrix{T}`).
2. The input image must be a `CUDA.CuArray` (CPU inputs fall back to eager execution).
3. Output coefficient buffers must have stable device pointers across calls
   (i.e., the caller must not allocate new coefficient containers each call).

## Compatibility Notes

- **NSCT workspace**: not yet captured (NSDFB kernel shape varies by subband level).
- **Other GPU backends** (AMDGPU, oneAPI, Metal): use `ContourletsGPUExt` directly;
  no graph capture — the workspace path still eliminates per-call allocations.
- **Filter cache**: only active during CUDA capture; outside capture,
  `_active_filter_cache()` returns `nothing` and `conv2d_sep!` calls `_ensure_gpu` normally.
