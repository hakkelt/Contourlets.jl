# The `threaded` Keyword

## Resolution (MultiscaleFilterBanks.jl, `src/threading.jl`)

`use_threading` and the FFT-loop task split (`_n_fft_tasks`, `_chunk`) live in
the shared core and are imported in `src/Contourlets.jl`.

```julia
@inline use_threading(threaded::Bool, ::Type) = threaded
@inline use_threading(::Nothing, ::Type{T}) where {T} = !(T <: Real)
```

`threaded = nothing` (the default) enables threading exactly when `T` is not a
real type — i.e., for all complex types. `Real` data takes the `@turbo` SIMD path
instead.

## Using `use_threading` in a Function

```julia
function my_filtered_op!(dst, src, h, threaded::Union{Bool, Nothing} = nothing)
    if use_threading(threaded, eltype(src))
        _my_op_batch!(dst, src, h)    # Polyester @batch loop
    else
        _my_op_turbo!(dst, src, h)    # LoopVectorization @turbo loop
    end
end
```

Call `use_threading` once at function entry; the branch costs nothing next to
the loops it selects.

## Propagating the Keyword

Always pass the `threaded` keyword through to inner calls that do hot work:

```julia
function nsct_forward!(coeffs, image, params; threaded = nothing)
    _nsct_inner!(coeffs, image, params, threaded)  # pass it down
end

function _nsct_inner!(coeffs, image, params, threaded::Union{Bool, Nothing})
    conv2d_sep!(dst, src, h, g; threaded = threaded)  # propagate
    ...
end
```

Never default to `nothing` in internal functions — always accept it as a parameter.
