@testitem "Documentation doctests" tags = [:docs, :quality] begin
    using Documenter
    DocMeta.setdocmeta!(Contourlets, :DocTestSetup, :(using Contourlets, Random); recursive = true)
    doctest(Contourlets; manual = false)
end

@testitem "Documentation structure (draft build)" tags = [:docs, :quality] begin
    using Documenter, MultiscaleFilterBanks
    pkg_root = pkgdir(Contourlets)
    # @meta blocks evaluate `CurrentModule = Contourlets` in Main; the module
    # must be importable there (same as `using Contourlets` in make.jl).
    Base.eval(Main, :(using Contourlets))
    DocMeta.setdocmeta!(
        MultiscaleFilterBanks, :DocTestSetup,
        :(using MultiscaleFilterBanks, Random); recursive = true
    )
    DocMeta.setdocmeta!(Contourlets, :DocTestSetup, :(using Contourlets, Random); recursive = true)
    build_dir = mktempdir()
    makedocs(;
        modules = [Contourlets, MultiscaleFilterBanks],
        remotes = Dict(
            pkgdir(MultiscaleFilterBanks) =>
                Documenter.Remotes.GitHub("hakkelt", "MultiscaleFilterBanks.jl"),
        ),
        sitename = "Contourlets.jl",
        root = joinpath(pkg_root, "docs"),
        source = "src",
        build = build_dir,
        checkdocs = :exports,
        draft = true,
        # Images referenced by `![](img.png)` are only generated when @example
        # blocks run (which draft=true skips), so suppress those link warnings.
        warnonly = [:cross_references],
    )
end
