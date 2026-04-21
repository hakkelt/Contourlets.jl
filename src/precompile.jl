using PrecompileTools

@setup_workload begin
    img = zeros(Float64, 32, 32)
    img[16, 16] = 1.0
    params = ContourletParams(J = 2, L_array = [1, 2])
    @compile_workload begin
        c = ct_forward(img, params)
        ct_inverse(c)
        nc = nsct_forward(img, params)
        nsct_inverse(nc)
        ws = make_workspace(params, (32, 32))
    end
end
