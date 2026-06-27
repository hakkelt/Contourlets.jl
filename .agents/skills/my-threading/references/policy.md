# ThreadingPolicy Dispatch

## Type Definitions (`src/threading.jl`)

```julia
abstract type ThreadingPolicy end
struct Auto    <: ThreadingPolicy end
struct Enabled <: ThreadingPolicy end
struct Disabled <: ThreadingPolicy end
```

## `_use_threading` Resolution

```julia
@inline _use_threading(::Enabled,  ::Type{T}) where {T} = true
@inline _use_threading(::Disabled, ::Type{T}) where {T} = false
@inline _use_threading(::Auto,     ::Type{T}) where {T} = !(T <: Real)
```

`Auto` enables threading exactly when `T` is not a real type — i.e., for all
complex types. `Real` data takes the `@turbo` SIMD path instead.

## Using `_use_threading` in a Function

```julia
function my_filtered_op!(dst, src, h, threading::ThreadingPolicy = Auto())
    if _use_threading(threading, eltype(src))
        _my_op_batch!(dst, src, h)    # Polyester @batch loop
    else
        _my_op_turbo!(dst, src, h)    # LoopVectorization @turbo loop
    end
end
```

The branch is `@inline`-ed and the policy type is concrete at the call site, so
the compiler eliminates the dead branch entirely — zero runtime overhead.

## Propagating the Policy

Always thread the `threading` kwarg through to inner calls that do hot work:

```julia
function ct_forward!(coeffs, image, params; threading=Auto())
    _ct_inner!(coeffs, image, params, threading)  # pass it down
end

function _ct_inner!(coeffs, image, params, threading::ThreadingPolicy)
    conv2d_sep!(dst, src, h, g; threading=threading)  # propagate
    ...
end
```

Never default to `Auto()` in internal functions — always accept it as a parameter.
