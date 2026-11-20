@testitem "interface conformance: real data" tags = [:ct, :nsct, :tict] begin
    using MultiscaleTransformsBase.Testing: test_transform
    using Random
    for t in (CT(J = 2, L_array = [2, 3]), NSCT(J = 2, L_array = [2, 3]), TICT(J = 2, L_array = [2, 1]), STICT(J = 2, L_array = [3, 2]))
        test_transform(t, randn(Xoshiro(1), 64, 64); wrong_size = (32, 32))
        test_transform(t, randn(Xoshiro(2), Float32, 64, 64); wrong_size = (32, 32))
    end
end

# Complex data runs through complex FFTs and real data through real ones, so
# real-linearity holds to rounding (`exact_real_linear = false`).
@testitem "interface conformance: complex data" tags = [:ct, :nsct, :tict] begin
    using MultiscaleTransformsBase.Testing: test_transform
    using Random
    for t in (CT(J = 2, L_array = [2, 3]), NSCT(J = 2, L_array = [2, 3]), TICT(J = 2, L_array = [2, 1]), STICT(J = 2, L_array = [3, 2]))
        test_transform(t, randn(Xoshiro(3), ComplexF64, 64, 64); wrong_size = (32, 32))
        test_transform(t, randn(Xoshiro(4), ComplexF32, 64, 64))
    end
end

@testitem "interface conformance: the interface is re-exported" tags = [:quality] begin
    using MultiscaleTransformsBase: MultiscaleTransformsBase
    for name in names(MultiscaleTransformsBase)
        (name === :MultiscaleTransformsBase || !Base.isexported(MultiscaleTransformsBase, name)) && continue
        @test Base.isexported(Contourlets, name)
        @test getfield(Contourlets, name) === getfield(MultiscaleTransformsBase, name)
    end
    @test CT <: AbstractMultiscaleTransform
    @test CTWorkspace <: AbstractTransformWorkspace
end
