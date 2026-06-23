using PrecompileTools

@setup_workload begin
    img = zeros(Float64, 32, 32)
    img[16, 16] = 1.0
    params = ContourletParams(J = 2, L_array = [1, 2])
    @compile_workload begin
        # Allocating CT / NSCT round trips.
        c = ct_forward(img, params)
        ct_inverse(c, params)
        nc = nsct_forward(img, params)
        nsct_inverse(nc, params)

        # In-place (workspace) paths — the iterative-algorithm entry points.
        ws = make_workspace(params, size(img))
        coeffs = similar_coefficients(params, size(img))
        ct_forward!(coeffs, img, ws)
        rec = similar(img)
        ct_inverse!(rec, coeffs, ws)

        nsws = make_nsct_workspace(params, size(img))
        ncoeffs = similar_nsct_coefficients(params, size(img))
        nsct_forward!(ncoeffs, img, nsws)
        nsct_inverse!(rec, ncoeffs, nsws)
    end
end
