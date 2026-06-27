# Writing Threaded Loops

## Real Data: `@turbo` (LoopVectorization)

```julia
using LoopVectorization

function _conv_turbo!(dst::AbstractMatrix{T}, src, h) where {T <: Real}
    n1, n2 = size(src)
    lh = length(h)
    @turbo for j in 1:n2, i in 1:n1   # column-major: j outer, i inner
        acc = zero(T)
        for k in 1:lh
            # ... boundary-safe index ...
            acc += h[k] * src[ii, j]
        end
        dst[i, j] = acc
    end
    return dst
end
```

Rules:
- `@turbo` requires all indexed arrays to be `Array` or `SubArray` (not arbitrary `AbstractArray`).
- Only `Real` element types — never use on complex data.
- Keep the inner body simple: no function calls that aren't inlineable, no `if` branches inside the vectorised nest that depend on loop indices in a non-trivial way.
- Column-major traversal: outer loop over `j`, inner over `i`.

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
- `@batch` parallelises the *outermost* loop only — put the largest-trip-count dimension there.
- For 2-D arrays use `j` (columns) as the outer loop.
- Polyester uses a persistent thread pool — no task allocation overhead; safe to call from inside another `@batch` region (it will not spawn more threads than `Threads.nthreads()`).
- Do not mix `@batch` and `@turbo` in the same function body.

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

Used when `threading = Disabled()` or when benchmarking threading overhead.
