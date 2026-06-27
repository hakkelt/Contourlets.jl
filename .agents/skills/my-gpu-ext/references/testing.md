# Testing the GPU Extension

## Test File

`test/items/test_gpu.jl` — all GPU tests carry the `:gpu` tag.

## Test Runner Setup (GPUEnv.jl + JLArrays)

CI runs use JLArrays, a pure-Julia GPU array simulator that exercises the
KernelAbstractions dispatch path without requiring real hardware:

```julia
@testitem "my GPU op" tags=[:gpu] begin
    using GPUEnv, JLArrays
    x = jl(rand(Float32, 64, 64))
    result = my_gpu_op(x)
    @test result isa JLArray
    @test Array(result) ≈ cpu_reference atol=1f-5
end
```

On machines with a real GPU backend (CUDA, AMDGPU, etc.), the same `@testitem`
runs against the real device because GPUEnv.jl picks the available backend.

## Running GPU Tests

```bash
# Run only GPU tests
julia --project=test test/runtests.jl :gpu

# GPU + CT correctness
julia --project=test test/runtests.jl :gpu,:ct
```

## Correctness Invariants to Test

- **Round-trip**: `ct_inverse(ct_forward(x, params), params) ≈ x` to Float32 tolerance (`1f-4`).
- **CPU/GPU match**: `Array(ct_forward(jl(x), params).coarse) ≈ ct_forward(x, params).coarse`.
- **Complex data**: repeat with `ComplexF32` input; result should satisfy `T(x+iy) = T(x) + i·T(y)`.
- **Adapt round-trip**: `Adapt.adapt(Array, Adapt.adapt(jl, coeffs_cpu))` reconstructs correctly.

## Common Pitfalls

- **Forgot `_scratch_like`**: using `zeros(T, m, n)` silently creates a CPU buffer; the kernel then fails with a type mismatch.
- **Index-vector gather**: `A[[2:end; 1], :]` triggers scalar indexing on JLArrays — replace with `circshift!`.
- **Filter on wrong device**: pass `_ensure_gpu(backend, h)` before the kernel; raw CPU vectors error on JLArrays.
- **QFB ladder path**: `qfb_decompose` does not yet support ladder-mode (`Q2345`) filters on GPU — it throws `ArgumentError`. Tests should use modulation-mode pairs or the DFB path (which handles ladder via the CPU `_ladder_modulate`).
