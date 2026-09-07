using PrecompileTools: @setup_workload, @compile_workload

@setup_workload begin
    img = zeros(Float64, 32, 32)
    img[16, 16] = 1.0
    params = ContourletParams(J = 2, L_array = [1, 2])
    # 3-D workload: small volume, low Bernstein order, and FFTW.ESTIMATE — a
    # MEASURE plan would time transforms during precompilation, and the wisdom it
    # produced would be for a size nobody asked for.
    vol = zeros(Float64, 8, 8, 8)
    vol[4, 4, 4] = 1.0
    params3d = ContourletParams3D(J = 1, L_array = [1], order = 4)
    @compile_workload begin
        # Allocating CT / NSCT round trips.
        c = ct_forward(img, params)
        ct_inverse(c, params)
        nc = nsct_forward(img, params)
        nsct_inverse(nc, params)

        # In-place (workspace) paths — the iterative-algorithm entry points.
        ws = make_workspace(params, size(img))
        coeffs = similar_coefficients(params, size(img))
        ct_forward!(coeffs, img, params; workspace = ws)
        rec = similar(img)
        ct_inverse!(rec, coeffs, params; workspace = ws)

        nsws = make_nsct_workspace(params, size(img))
        ncoeffs = similar_nsct_coefficients(params, size(img))
        nsct_forward!(ncoeffs, img, params; workspace = nsws)
        nsct_inverse!(rec, ncoeffs, params; workspace = nsws)

        # 3-D: the allocating round trip, the workspace path and the streaming
        # form, which is the entry point at working sizes.
        ws3d = make_nsct3d_workspace(params3d, size(vol), Float64; flags = FFTW.ESTIMATE)
        c3d = nsct3d_forward(vol, params3d; workspace = ws3d)
        rec3d = similar(vol)
        nsct3d_inverse!(rec3d, c3d, params3d; workspace = ws3d)
        nsct3d_forward(vol, params3d; workspace = ws3d) do j, k, band
            sum(band)
        end
    end
end
