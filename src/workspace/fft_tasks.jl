# The threaded subband loop of the DFT-mask workspaces (NSCT, TICT).
#
# Their outputs are independent: each is a pointwise product with the input
# spectrum and one inverse FFT.  So the outputs are split into contiguous chunks,
# one task per FFT buffer (`_n_tasks`), each with its own buffers and, for the
# synthesis, its own partial sum (`_chunked_sum!`).  Both come from
# MultiscaleFilterBanks; this package runs the chunks with Polyester's `@batch`,
# which allocates about 250 B per call where `Threads.@threads` allocates ~4.5 KB,
# and the workspace paths promise (almost) no allocation.  Plans stay
# single-threaded (`_ns_plans`).

# Call `f(range, t)` on the chunks of `1:n`, one per task `t` in `1:nt`, threaded
# when `threaded`; otherwise call `f(1:n, 1)` once.  `_foreach_chunk`'s contract.
function _batch_chunks(f::F, n::Int, nt::Int, threaded::Bool) where {F}
    if threaded && nt > 1
        @batch for t in 1:nt
            f(_chunk(n, nt, t), t)
        end
    else
        f(1:n, 1)
    end
    return nothing
end
