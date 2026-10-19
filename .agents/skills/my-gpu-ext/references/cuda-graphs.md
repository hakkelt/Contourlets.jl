# CUDA Graph Capture (ContourletsCUDAExt)

## What It Does

The workspace paths of the CT (`_ct_forward_ws!`, `_ct_inverse_ws!`) and the STICT
(`_stict_forward_ws!`, `_stict_inverse_ws!`) have a fixed kernel shape for a given
`(image_size, params, eltype)`. `ContourletsCUDAExt` captures each as a CUDA graph
and replays it with a single host call, replacing ~300 (CT) or ~1,700 (STICT)
individual kernel launches. All four go through one helper, `_replay_or_capture!`.

## Capture Protocol (`_replay_or_capture!`)

**Cache:** each path keeps up to `_GRAPH_CACHE_SIZE` (4) graphs in a `_GraphCache`,
most recently used first, each keyed on the device addresses of the input, the
lowpass and *every* subband (a new coefficient set may reuse some of the old
addresses but not all). A solver alternating between coefficient sets replays.

**Hit:** `CUDA.launch(exec)`, a single host call; the comparison allocates nothing.

**Miss:**
- Run the path eagerly once. It writes a valid result and, for the CT, populates
  all `ScratchArena` slots, so their device pointers are stable before capture.
- `prepare(f)` runs outside the capture and returns `(g, keep)`: what to capture,
  and the device arrays the graph reads that nothing else roots (a graph does not
  keep them alive). The entry holds `keep`, never `g`, which holds the caller's
  arrays. For the CT (`_ct_prepare`) it pre-builds the device LP filter vectors
  (`CuArray{Tf}(fp.h)`; `cu` would make Float64 filters Float32)
  and resets the arena cursor (`_arena_reset!`); the captured call runs inside
  `_with_filter_cache`, so `conv2d_sep!` finds the stable vectors instead of calling
  `_ensure_gpu` (which would create graph-owned memory nodes that fail to free via
  the normal pool). The STICT's buffers and plans are built with its workspace, so
  it captures the path as it is.
- Capture, instantiate, and evict the least recently used graph past the limit.

**Stale buffers:** a CT call with another size replaces `ScratchArena` buffers the
cached graphs point at. The arena counts replacements (`replaced`), the cache
stores the count it captured with (`epoch`), and a change drops every graph.

**Allocating calls** (`ct_forward(…; workspace)`, `stict_forward`, …) run eagerly:
their arrays are new on every call, so a graph would never replay and would evict
the ones an in-place solver relies on.

## Preconditions for Capture to Trigger

1. The workspace must be CUDA-resident (`M <: CUDA.CuMatrix{T}`).
2. The input image must be a `CUDA.CuArray` (CPU inputs fall back to eager execution).
3. Output coefficient buffers must have stable device pointers across calls
   (i.e., the caller must not allocate new coefficient containers each call).

## Compatibility Notes

- **NSCT / TICT workspaces**: not captured. On a device their analysis runs in chunks
  of subbands with batched FFTs (MultiscaleFilterBanks' `MaskBankEngine` with a
  `planner`), which already cuts the launches several-fold.
- **Other GPU backends** (AMDGPU, oneAPI, Metal): use `ContourletsGPUExt` directly;
  no graph capture — the workspace path still eliminates per-call allocations.
- **Filter cache**: only active during CUDA capture; outside capture,
  `_active_filter_cache()` returns `nothing` and `conv2d_sep!` calls `_ensure_gpu` normally.
