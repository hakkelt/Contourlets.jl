#!/usr/bin/env julia
#
# GPU vs CPU benchmark for the Contourlet transforms.
#
# USAGE:
#     julia --project=benchmark benchmark/gpu_benchmark.jl
#
# GPUEnv builds an overlay environment with the GPU stack (GPUArrays,
# KernelAbstractions and any functional native backend) and activates it, so the
# ContourletsGPUExt extension loads automatically.  The transforms run the
# Laplacian/nonsubsampled pyramid stage on the device and the directional stage
# on the host (see ext/ContourletsGPUExt/transforms_gpu.jl).

using BenchmarkTools
using Contourlets
using GPUEnv
using Printf
using Random

GPUEnv.activate(persist = true, include_jlarrays = false, only_first = true)

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
    p = ContourletParams(
        J = 3, L_array = parabolic_levels(3),
        lp_filters = FilterPair{Float32}(Float32.(CDF97.h), Float32.(CDF97.g)),
        dfb_filters = QuincunxFilterPair{Float32}(
            Float32.(Q2345.h_q), Float32.(Q2345.g_q),
            Q2345.c_h, Q2345.c_g, Float32.(Q2345.f_ladder)
        )
    )

    ws_cpu = make_workspace(x, p)
    ws_gpu = make_workspace(xg, p)
    c_cpu_alloc = similar_coefficients(p, (n, n), Td = Float32, M = typeof(x))
    c_gpu_alloc = similar_coefficients(p, (n, n), Td = Float32, M = typeof(xg))

    benchmark_case("ct forward", "$(n)x$(n)", () -> ct_forward(x, p), () -> ct_forward(xg, p))
    benchmark_case("ct forward!", "$(n)x$(n)", () -> ct_forward!(c_cpu_alloc, x, ws_cpu), () -> ct_forward!(c_gpu_alloc, xg, ws_gpu))
end

print_header("Contourlet Transform — inverse")
for n in SIZES
    x = randn(Float32, n, n)
    xg = to_gpu(GPU_BACKEND, x)
    p = ContourletParams(
        J = 3, L_array = parabolic_levels(3),
        lp_filters = FilterPair{Float32}(Float32.(CDF97.h), Float32.(CDF97.g)),
        dfb_filters = QuincunxFilterPair{Float32}(
            Float32.(Q2345.h_q), Float32.(Q2345.g_q),
            Q2345.c_h, Q2345.c_g, Float32.(Q2345.f_ladder)
        )
    )

    ws_cpu = make_workspace(x, p)
    ws_gpu = make_workspace(xg, p)

    c_cpu = ct_forward(x, p)
    c_gpu = ct_forward(xg, p)

    x_out_cpu = similar(x)
    x_out_gpu = similar(xg)

    benchmark_case(
        "ct inverse", "$(n)x$(n)",
        () -> ct_inverse(c_cpu),
        () -> ct_inverse(c_gpu, xg),
    )
    benchmark_case(
        "ct inverse!", "$(n)x$(n)",
        () -> ct_inverse!(x_out_cpu, c_cpu, ws_cpu),
        () -> ct_inverse!(x_out_gpu, c_gpu, ws_gpu),
    )
end

print_header("Nonsubsampled Contourlet Transform — forward")
for n in SIZES
    x = randn(Float32, n, n)
    xg = to_gpu(GPU_BACKEND, x)
    p = ContourletParams(
        J = 2, L_array = [2, 3],
        lp_filters = FilterPair{Float32}(Float32.(CDF97.h), Float32.(CDF97.g)),
        dfb_filters = QuincunxFilterPair{Float32}(
            Float32.(Q2345.h_q), Float32.(Q2345.g_q),
            Q2345.c_h, Q2345.c_g, Float32.(Q2345.f_ladder)
        )
    )

    ws_cpu = make_nsct_workspace(x, p)
    ws_gpu = make_nsct_workspace(xg, p)
    c_cpu_alloc = similar_nsct_coefficients(p, (n, n), Td = Float32, M = typeof(x))
    c_gpu_alloc = similar_nsct_coefficients(p, (n, n), Td = Float32, M = typeof(xg))

    benchmark_case("nsct forward", "$(n)x$(n)", () -> nsct_forward(x, p), () -> nsct_forward(xg, p))
    benchmark_case("nsct forward!", "$(n)x$(n)", () -> nsct_forward!(c_cpu_alloc, x, ws_cpu), () -> nsct_forward!(c_gpu_alloc, xg, ws_gpu))
end

print_header("Nonsubsampled Contourlet Transform — inverse")
for n in SIZES
    x = randn(Float32, n, n)
    xg = to_gpu(GPU_BACKEND, x)
    p = ContourletParams(
        J = 2, L_array = [2, 3],
        lp_filters = FilterPair{Float32}(Float32.(CDF97.h), Float32.(CDF97.g)),
        dfb_filters = QuincunxFilterPair{Float32}(
            Float32.(Q2345.h_q), Float32.(Q2345.g_q),
            Q2345.c_h, Q2345.c_g, Float32.(Q2345.f_ladder)
        )
    )

    ws_cpu = make_nsct_workspace(x, p)
    ws_gpu = make_nsct_workspace(xg, p)

    c_cpu = nsct_forward(x, p)
    c_gpu = nsct_forward(xg, p)

    x_out_cpu = similar(x)
    x_out_gpu = similar(xg)

    benchmark_case(
        "nsct inverse", "$(n)x$(n)",
        () -> nsct_inverse(c_cpu),
        () -> nsct_inverse(c_gpu, xg),
    )
    benchmark_case(
        "nsct inverse!", "$(n)x$(n)",
        () -> nsct_inverse!(x_out_cpu, c_cpu, ws_cpu),
        () -> nsct_inverse!(x_out_gpu, c_gpu, ws_gpu),
    )
end

println()
println(repeat("=", 78))
println("Summary")
println(repeat("=", 78))
for (name, vals) in sort(collect(results), by = first)
    @printf("%-18s\t avg: %6.2fx, min: %6.2fx, max: %6.2fx\n", name, sum(vals) / length(vals), minimum(vals), maximum(vals))
end
