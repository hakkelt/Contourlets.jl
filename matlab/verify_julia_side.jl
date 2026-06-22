# Julia side of the MATLAB↔Julia CT/NSCT/NSDFB cross-validation.
#
# Reads shared inputs written by verify_vs_julia.m, runs Julia's NSDFB,
# CT, and NSCT at each requested size, writes subbands / reconstructions /
# timings as CSVs that the MATLAB script reads back for comparison.
#
# Usage:
#   julia --project=<pkg> verify_julia_side.jl <io_dir>

using Contourlets, DelimitedFiles, Printf

iodir = abspath(ARGS[1])

# ── Parameters ────────────────────────────────────────────────────────────────

sizes_raw = readdlm(joinpath(iodir, "sizes.csv"), ',', Float64)
sizes = Int.(vec(sizes_raw))

J = Int(readdlm(joinpath(iodir, "J.csv"), ',', Float64)[1])
L_raw = vec(readdlm(joinpath(iodir, "L_array.csv"), ',', Float64))
L_array = Int.(L_raw)
L_nsdfb = Int(readdlm(joinpath(iodir, "L_nsdfb.csv"), ',', Float64)[1])

params = ContourletParams(J = J, L_array = L_array)

# ── Timing helper ─────────────────────────────────────────────────────────────

const NRUN = 7

function timeit(f)
    f()            # compile + warm up
    ts = Float64[]
    for _ in 1:NRUN
        push!(ts, @elapsed f())
    end
    return sort(ts)[(NRUN + 1) ÷ 2]   # median
end

# ── Per-size loop ─────────────────────────────────────────────────────────────

timing_rows = Tuple{String, Int, Float64, Float64}[]

for n in sizes
    x = readdlm(joinpath(iodir, "x_$(n).csv"), ',', Float64)

    # ── NSDFB (standalone, no pyramid) ────────────────────────────────────────
    sbs_nsdfb = nsdfb_decompose(x, L_nsdfb, Q2345, 1)
    for k in eachindex(sbs_nsdfb)
        writedlm(joinpath(iodir, @sprintf("jl_nsdfb_%d_sb%02d.csv", n, k)), sbs_nsdfb[k], ',')
    end
    rec_nsdfb = nsdfb_reconstruct(sbs_nsdfb, Q2345, 1)
    writedlm(joinpath(iodir, @sprintf("jl_nsdfb_%d_rec.csv", n)), rec_nsdfb, ',')

    td_nsdfb = timeit(() -> nsdfb_decompose(x, L_nsdfb, Q2345, 1))
    tr_nsdfb = timeit(() -> nsdfb_reconstruct(sbs_nsdfb, Q2345, 1))
    push!(timing_rows, ("nsdfb", n, td_nsdfb, tr_nsdfb))
    @printf("[julia] NSDFB  n=%-4d  L=%d  dec %.4f s  rec %.4f s\n", n, L_nsdfb, td_nsdfb, tr_nsdfb)

    # ── CT ────────────────────────────────────────────────────────────────────
    coeffs_ct = ct_forward(x, params)
    for j in 1:J, k in eachindex(coeffs_ct.subbands[j])
        writedlm(joinpath(iodir, @sprintf("jl_ct_%d_j%d_sb%02d.csv", n, j, k)), coeffs_ct.subbands[j][k], ',')
    end
    writedlm(joinpath(iodir, @sprintf("jl_ct_%d_coarse.csv", n)), coeffs_ct.coarse, ',')
    rec_ct = ct_inverse(coeffs_ct)
    writedlm(joinpath(iodir, @sprintf("jl_ct_%d_rec.csv", n)), rec_ct, ',')

    td_ct = timeit(() -> ct_forward(x, params))
    tr_ct = timeit(() -> ct_inverse(coeffs_ct))
    push!(timing_rows, ("ct", n, td_ct, tr_ct))
    @printf("[julia] CT     n=%-4d  J=%d  L=%s  dec %.4f s  rec %.4f s\n", n, J, L_array, td_ct, tr_ct)

    # ── NSCT (allocating, spatial kernels) ────────────────────────────────────
    coeffs_nsct = nsct_forward(x, params)
    for j in 1:J, k in eachindex(coeffs_nsct.subbands[j])
        writedlm(joinpath(iodir, @sprintf("jl_nsct_%d_j%d_sb%02d.csv", n, j, k)), coeffs_nsct.subbands[j][k], ',')
    end
    writedlm(joinpath(iodir, @sprintf("jl_nsct_%d_coarse.csv", n)), coeffs_nsct.coarse, ',')
    rec_nsct = nsct_inverse(coeffs_nsct)
    writedlm(joinpath(iodir, @sprintf("jl_nsct_%d_rec.csv", n)), rec_nsct, ',')

    td_nsct = timeit(() -> nsct_forward(x, params))
    tr_nsct = timeit(() -> nsct_inverse(coeffs_nsct))
    push!(timing_rows, ("nsct", n, td_nsct, tr_nsct))
    @printf("[julia] NSCT   n=%-4d  J=%d  L=%s  dec %.4f s  rec %.4f s\n", n, J, L_array, td_nsct, tr_nsct)

    # ── NSCT workspace (FFT path) ─────────────────────────────────────────────
    ws = make_nsct_workspace(params, (n, n))
    coeffs_ws = similar_nsct_coefficients(params, (n, n))
    nsct_forward!(coeffs_ws, x, ws)

    td_nsct_ws = timeit(() -> nsct_forward!(coeffs_ws, x, ws))
    tr_nsct_ws = timeit(() -> nsct_inverse!(similar(x), coeffs_ws, ws))
    push!(timing_rows, ("nsct_ws", n, td_nsct_ws, tr_nsct_ws))
    @printf("[julia] NSCT_WS n=%-4d J=%d  dec %.4f s  rec %.4f s\n", n, J, td_nsct_ws, tr_nsct_ws)
end

# ── Write timing table ────────────────────────────────────────────────────────
open(joinpath(iodir, "jl_timing.csv"), "w") do io
    println(io, "transform,size,decompose_s,reconstruct_s")
    for (tr, n, td, trr) in timing_rows
        println(io, "$(tr),$(n),$(td),$(trr)")
    end
end
println("[julia] Done.  Results written to $(iodir)")
