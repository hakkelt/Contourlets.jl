#!/usr/bin/env julia
#
# GPU vs CPU benchmark for the Contourlet transforms.
#
# USAGE:
#     julia --project=benchmark benchmark/gpu_benchmark.jl
#
# GPUEnv builds an overlay environment with the GPU stack (GPUArrays,
# KernelAbstractions and any functional native backend) and activates it, so the
# ContourletsGPUExt extension loads automatically.  Every stage of the
# transforms runs on the device.

using BenchmarkTools
using Contourlets
using GPUEnv
using Printf
using Random

GPUEnv.activate(persist = true, include_cpu_backends = false, only_first = true)

const BACKENDS = gpu_backends()

if isempty(BACKENDS)
    println("No functional native GPU backend detected. Skipping GPU benchmark.")
    exit(0)
end

const GPU_BACKEND = first(BACKENDS)
Random.seed!(0)

function bench_time(f; samples = 10, evals = 1)
    trial = run(
        @benchmarkable begin
            local result = $f()
            synchronize_backend(GPU_BACKEND)
            result
        end samples = samples evals = evals
    )
    return median(trial.times) / 1.0e9
end

function print_header(title)
    println()
    println(repeat("=", 78))
    println(title)
    println(repeat("=", 78))
    return @printf("%-16s %12s %12s %10s\n", "Size", "CPU (ms)", "GPU (ms)", "Speedup")
end

function print_row(label, cpu_t, gpu_t)
    speedup = cpu_t / gpu_t
    @printf("%-16s %12.3f %12.3f %10.2fx\n", label, cpu_t * 1.0e3, gpu_t * 1.0e3, speedup)
    return speedup
end

results = Dict{String, Vector{Float64}}()
record!(k, v) = push!(get!(results, k, Float64[]), v)

function benchmark_case(group, label, cpu_f, gpu_f)
    cpu_t = bench_time(cpu_f)
    gpu_t = bench_time(gpu_f)
    return record!(group, print_row(label, cpu_t, gpu_t))
end

println("Contourlets.jl GPU benchmark")
println("Backend: ", GPU_BACKEND.name)

const SIZES = (128, 256, 512)

print_header("Contourlet Transform — forward")
for n in SIZES
    x = randn(Float32, n, n)
    xg = to_gpu(GPU_BACKEND, x)
    t = CT(J = 3, L_array = parabolic_levels(3))

    ws_cpu = make_workspace(t, x)
    ws_gpu = make_workspace(t, xg)
    c_cpu_alloc = similar_coefficients(t, x)
    c_gpu_alloc = similar_coefficients(t, xg)

    benchmark_case("ct forward", "$(n)x$(n)", () -> analyze(t, x), () -> analyze(t, xg))
    benchmark_case("ct forward!", "$(n)x$(n)", () -> analyze!(c_cpu_alloc, t, x; workspace = ws_cpu), () -> analyze!(c_gpu_alloc, t, xg; workspace = ws_gpu))
end

print_header("Contourlet Transform — inverse")
for n in SIZES
    x = randn(Float32, n, n)
    xg = to_gpu(GPU_BACKEND, x)
    t = CT(J = 3, L_array = parabolic_levels(3))

    ws_cpu = make_workspace(t, x)
    ws_gpu = make_workspace(t, xg)

    c_cpu = analyze(t, x)
    c_gpu = analyze(t, xg)

    x_out_cpu = similar(x)
    x_out_gpu = similar(xg)

    benchmark_case(
        "ct inverse", "$(n)x$(n)",
        () -> synthesize(t, c_cpu),
        () -> synthesize(t, c_gpu),
    )
    benchmark_case(
        "ct inverse!", "$(n)x$(n)",
        () -> synthesize!(x_out_cpu, t, c_cpu; workspace = ws_cpu),
        () -> synthesize!(x_out_gpu, t, c_gpu; workspace = ws_gpu),
    )
end

print_header("Nonsubsampled Contourlet Transform — forward")
for n in SIZES
    x = randn(Float32, n, n)
    xg = to_gpu(GPU_BACKEND, x)
    t = NSCT(J = 2, L_array = [2, 3])

    ws_cpu = make_workspace(t, x)
    ws_gpu = make_workspace(t, xg)
    c_cpu_alloc = similar_coefficients(t, x)
    c_gpu_alloc = similar_coefficients(t, xg)

    benchmark_case("nsct forward", "$(n)x$(n)", () -> analyze(t, x), () -> analyze(t, xg))
    benchmark_case("nsct forward!", "$(n)x$(n)", () -> analyze!(c_cpu_alloc, t, x; workspace = ws_cpu), () -> analyze!(c_gpu_alloc, t, xg; workspace = ws_gpu))
end

print_header("Nonsubsampled Contourlet Transform — inverse")
for n in SIZES
    x = randn(Float32, n, n)
    xg = to_gpu(GPU_BACKEND, x)
    t = NSCT(J = 2, L_array = [2, 3])

    ws_cpu = make_workspace(t, x)
    ws_gpu = make_workspace(t, xg)

    c_cpu = analyze(t, x)
    c_gpu = analyze(t, xg)

    x_out_cpu = similar(x)
    x_out_gpu = similar(xg)

    benchmark_case(
        "nsct inverse", "$(n)x$(n)",
        () -> synthesize(t, c_cpu),
        () -> synthesize(t, c_gpu),
    )
    benchmark_case(
        "nsct inverse!", "$(n)x$(n)",
        () -> synthesize!(x_out_cpu, t, c_cpu; workspace = ws_cpu),
        () -> synthesize!(x_out_gpu, t, c_gpu; workspace = ws_gpu),
    )
end

println()
println(repeat("=", 78))
println("Summary")
println(repeat("=", 78))
for (name, vals) in sort(collect(results), by = first)
    @printf("%-18s\t avg: %6.2fx, min: %6.2fx, max: %6.2fx\n", name, sum(vals) / length(vals), minimum(vals), maximum(vals))
end
