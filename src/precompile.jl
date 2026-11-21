using PrecompileTools: @setup_workload, @compile_workload

@setup_workload begin
    img = zeros(Float64, 32, 32)
    img[16, 16] = 1.0
    ct = CT(scales = 2, direction_levels = [1, 2])
    nsct = NSCT(scales = 2, direction_levels = [1, 2])
    @compile_workload begin
        # Allocating CT / NSCT round trips.
        synthesize(ct, analyze(ct, img))
        synthesize(nsct, analyze(nsct, img))

        # In-place workspace paths — the iterative-algorithm entry points.
        rec = similar(img)
        for t in (ct, nsct)
            ws = make_workspace(t, img)
            c = similar_coefficients(t, img)
            analyze!(c, t, img; workspace = ws)
            synthesize!(rec, t, c; workspace = ws)
        end
    end
end
