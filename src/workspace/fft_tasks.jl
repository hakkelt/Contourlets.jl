# The threaded subband loop of the DFT-mask workspaces (NSCT, TICT).
#
# Their outputs are independent: each is a pointwise product with the input
# spectrum and one inverse FFT.  So the outputs are split into contiguous chunks,
# one task per FFT buffer (`_n_fft_tasks`), each with its own buffers and, for
# the synthesis, its own partial sum.  Plans stay single-threaded (`_ns_plans`).

# Number of tasks of a workspace with `n` outputs on arrays like `X`.
_n_tasks(X::AbstractArray, n::Int) = max(1, min(_n_fft_tasks(X), n))

# Call `f(range, t)` on the chunks of `1:n`, one per task `t` in `1:nt`, threaded
# when `threaded`; otherwise call `f(1:n, 1)` once.
function _foreach_chunk(f::F, n::Int, nt::Int, threaded::Bool) where {F}
    if threaded && nt > 1
        @batch for t in 1:nt
            f(_chunk(n, nt, t), t)
        end
    else
        f(1:n, 1)
    end
    return nothing
end

# acc += the sum of `f(a, range, t)` over the chunks, where task 1 adds into `acc`
# and task t ≥ 2 into its own zeroed partial sum `partials[t - 1]`.
function _chunked_sum!(f::F, acc::AbstractArray, partials::AbstractVector, n::Int, threaded::Bool) where {F}
    nt = length(partials) + 1
    _foreach_chunk(n, nt, threaded) do range, t
        a = t == 1 ? acc : fill!(partials[t - 1], zero(eltype(acc)))
        f(a, range, t)
    end
    if threaded && nt > 1
        for a in partials
            acc .+= a
        end
    end
    return acc
end
