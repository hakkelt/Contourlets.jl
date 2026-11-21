@testitem "interface conformance: real data" tags = [:ct, :nsct, :tict] begin
    using MultiscaleTransformsBase.Testing: test_transform
    using Random
    for t in (CT(scales = 2, direction_levels = [3, 2]), NSCT(scales = 2, direction_levels = [3, 2]), TICT(scales = 2, direction_levels = [1, 2]), STICT(scales = 2, direction_levels = [2, 3]))
        test_transform(t, randn(Xoshiro(1), 64, 64); wrong_size = (32, 32))
        test_transform(t, randn(Xoshiro(2), Float32, 64, 64); wrong_size = (32, 32))
    end
end

# Complex data runs through complex FFTs and real data through real ones, so
# real-linearity holds to rounding (`exact_real_linear = false`).
@testitem "interface conformance: complex data" tags = [:ct, :nsct, :tict] begin
    using MultiscaleTransformsBase.Testing: test_transform
    using Random
    for t in (CT(scales = 2, direction_levels = [3, 2]), NSCT(scales = 2, direction_levels = [3, 2]), TICT(scales = 2, direction_levels = [1, 2]), STICT(scales = 2, direction_levels = [2, 3]))
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
