# Writing Threaded Loops

## Real Data: `@simd`

```julia
function _conv_simd!(dst::AbstractMatrix{T}, src, h) where {T <: Real}
    n1, n2 = size(src)
    lh = length(h)
    for j in 1:n2                      # column-major: j outer, i inner
        dst[:, j] .= zero(T)
        for k in 1:lh
            @simd for i in 1:n1        # stride-1 axpy over the column
                dst[i, j] += h[k] * src[i, j]   # ... boundary-safe index ...
            end
        end
    end
    return dst
end
```

Rules:
- Keep the innermost loop a stride-1 axpy ("zero, then accumulate over k") so it vectorizes.
- Column-major traversal: outer loop over `j`, inner over `i`.
- Split the interior (no boundary extension) from the border band, which takes the slower boundary-aware path.

## Complex Data: `@batch` (Polyester)

```julia
using Polyester

function _conv_batch!(dst::AbstractMatrix{T}, src, h) where {T <: Complex}
    n1, n2 = size(src)
    lh = length(h)
    @batch for j in 1:n2
        for i in 1:n1
            acc = zero(T)
            for k in 1:lh
                acc += h[k] * src[ii, j]
            end
            dst[i, j] = acc
        end
    end
    return dst
end
```

Rules:
- `@batch` parallelizes the *outermost* loop only — put the largest-trip-count dimension there.
- For 2-D arrays use `j` (columns) as the outer loop.
- Polyester uses a persistent thread pool — no task allocation overhead; safe to call from inside another `@batch` region (it will not spawn more threads than `Threads.nthreads()`).

## Disabled Path: Plain Loop

```julia
function _conv_plain!(dst, src, h)
    n1, n2 = size(src)
    lh = length(h)
    for j in 1:n2, i in 1:n1
        acc = zero(eltype(dst))
        for k in 1:lh
            acc += h[k] * src[ii, j]
        end
        dst[i, j] = acc
    end
end
```

Used when `threaded = false` or when benchmarking threading overhead.
