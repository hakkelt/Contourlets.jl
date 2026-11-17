# Adding Threading to a New Function

## Checklist

1. **Accept `threaded::Union{Bool, Nothing} = nothing` as a keyword argument** in the public-facing function.

2. **Propagate it** — pass `threaded` (not `nothing`) to every inner call that does compute work.

3. **Split into two implementation functions** — one single-threaded `@simd` loop (real) and one `@batch` (complex). Name them `_foo_simd!` and `_foo_batch!`.

4. **Dispatch via `use_threading`**:
   ```julia
   function foo!(dst, src, params; threaded = nothing)
       if use_threading(threaded, eltype(src))
           _foo_batch!(dst, src, params)
       else
           _foo_simd!(dst, src, params)
       end
   end
   ```

5. **Restrict the `@simd` method to real data** — `where {T <: Real}` on its signature, so complex data always takes the `@batch` method.

6. **Test both paths**:
   ```julia
   @test foo(x_real,    params; threaded = false) ≈ foo(x_real,    params; threaded = nothing)
   @test foo(x_complex, params; threaded = false) ≈ foo(x_complex, params; threaded = nothing)
   ```

7. **Benchmark both paths** — confirm `threaded = true` on real data is not faster than the default (it should be the same or slower due to task overhead vs SIMD).

## What NOT to Do

- Do not use `Base.Threads.@spawn` — task creation is too expensive for recursive transform paths.
- Do not use `Threads.@threads` — it does not integrate with the `threaded` keyword and has higher overhead than `Polyester`.
- Do not call `use_threading` inside a hot inner loop — call it once at the function entry and branch to the appropriate implementation.
