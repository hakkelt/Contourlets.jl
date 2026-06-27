# Adding Threading to a New Function

## Checklist

1. **Accept `threading::ThreadingPolicy = Auto()` as a keyword argument** in the public-facing function.

2. **Propagate it** — pass `threading` (not `Auto()`) to every inner call that does compute work.

3. **Split into two implementation functions** — one `@turbo` (real) and one `@batch` (complex), or a plain fallback. Name them `_foo_turbo!` and `_foo_batch!`.

4. **Dispatch via `_use_threading`**:
   ```julia
   function foo!(dst, src, params; threading=Auto())
       if _use_threading(threading, eltype(src))
           _foo_batch!(dst, src, params)
       else
           _foo_turbo!(dst, src, params)
       end
   end
   ```

5. **Guard `@turbo` with a type constraint** — `where {T <: Real}` on the method signature so the compiler never generates a `@turbo` specialisation for complex types.

6. **Test both paths**:
   ```julia
   @test foo(x_real,    params; threading=Disabled()) ≈ foo(x_real,    params; threading=Auto())
   @test foo(x_complex, params; threading=Disabled()) ≈ foo(x_complex, params; threading=Auto())
   ```

7. **Benchmark both paths** — confirm `Enabled()` on real data is not faster than `Auto()` (it should be the same or slower due to task overhead vs SIMD).

## What NOT to Do

- Do not use `Base.Threads.@spawn` — task creation is too expensive for recursive transform paths.
- Do not use `Threads.@threads` — it does not integrate with the `ThreadingPolicy` API and has higher overhead than `Polyester`.
- Do not call `_use_threading` inside a hot inner loop — call it once at the function entry and branch to the appropriate implementation.
- Do not put `@turbo` on a loop that reads from `AbstractMatrix` (requires `Array` or `SubArray` for SIMD to kick in — annotate `where {T <: Real}` and accept `Array{T,2}` internally if needed).
