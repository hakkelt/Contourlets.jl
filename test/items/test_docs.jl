@testitem "Documentation doctests" tags = [:docs, :quality] begin
    using Documenter
    DocMeta.setdocmeta!(Contourlets, :DocTestSetup, :(using Contourlets, Random); recursive = true)
    doctest(Contourlets; manual = false)
end

@testitem "Documentation structure (draft build)" tags = [:docs, :quality] begin
    using Documenter, MultiscaleFilterBanks, AbstractMultiscaleTransforms
    using Logging: Logging, with_logger
    pkg_root = pkgdir(Contourlets)
    # @meta blocks evaluate `CurrentModule = Contourlets` in Main; the module
    # must be importable there (same as `using Contourlets` in make.jl).
    Base.eval(Main, :(using Contourlets, AbstractMultiscaleTransforms))
    DocMeta.setdocmeta!(
        MultiscaleFilterBanks, :DocTestSetup,
        :(using MultiscaleFilterBanks, Random); recursive = true
    )
    DocMeta.setdocmeta!(
        AbstractMultiscaleTransforms, :DocTestSetup,
        :(using AbstractMultiscaleTransforms); recursive = true
    )
    DocMeta.setdocmeta!(Contourlets, :DocTestSetup, :(using Contourlets, Random); recursive = true)
    build_dir = mktempdir()
    logger = Test.TestLogger(; min_level = Logging.Warn)
    with_logger(logger) do
        makedocs(;
            modules = [Contourlets, MultiscaleFilterBanks, AbstractMultiscaleTransforms],
            checkdocs_ignored_modules = [MultiscaleFilterBanks, AbstractMultiscaleTransforms],
            remotes = Dict(
                pkgdir(MultiscaleFilterBanks) =>
                    Documenter.Remotes.GitHub("hakkelt", "MultiscaleFilterBanks.jl"),
                pkgdir(AbstractMultiscaleTransforms) =>
                    Documenter.Remotes.GitHub("hakkelt", "AbstractMultiscaleTransforms.jl"),
            ),
            sitename = "Contourlets.jl",
            root = joinpath(pkg_root, "docs"),
            source = "src",
            build = build_dir,
            checkdocs = :public,
            draft = true,
            # Images referenced by `![](img.png)` are only generated when @example
            # blocks run (which draft=true skips): their link warnings are expected,
            # and filtered out below; any other cross-reference warning fails.
            warnonly = [:cross_references],
        )
    end
    missing_image(r) = occursin("invalid local link/image", string(r.message)) &&
        any(v -> occursin(".png", repr(v)), values(r.kwargs))
    cross_reference(r) = occursin("@ref", string(r.message)) || occursin("local link", string(r.message))
    unexpected = [string(r.message) for r in logger.logs if cross_reference(r) && !missing_image(r)]
    @test isempty(unexpected)
end
