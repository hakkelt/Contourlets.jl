using Documenter, Contourlets, MultiscaleFilterBanks, MultiscaleTransformsBase

DocMeta.setdocmeta!(Contourlets, :DocTestSetup, :(using Contourlets, Random); recursive = true)
DocMeta.setdocmeta!(
    MultiscaleFilterBanks, :DocTestSetup,
    :(using MultiscaleFilterBanks, Random); recursive = true
)
DocMeta.setdocmeta!(
    MultiscaleTransformsBase, :DocTestSetup,
    :(using MultiscaleTransformsBase); recursive = true
)

makedocs(;
    # MultiscaleFilterBanks and MultiscaleTransformsBase are in the list because
    # Contourlets re-exports names whose docstrings live there (`FilterPair`,
    # `CDF97`; the interface).  Their other public names are documented in their
    # own manuals, so `checkdocs` skips them (Documenter ≥ 1.20 for listed modules).
    modules = [Contourlets, MultiscaleFilterBanks, MultiscaleTransformsBase],
    checkdocs_ignored_modules = [MultiscaleFilterBanks, MultiscaleTransformsBase],
    sitename = "Contourlets.jl",
    remotes = Dict(
        pkgdir(MultiscaleFilterBanks) =>
            Documenter.Remotes.GitHub("hakkelt", "MultiscaleFilterBanks.jl"),
        pkgdir(MultiscaleTransformsBase) =>
            Documenter.Remotes.GitHub("hakkelt", "MultiscaleTransformsBase.jl"),
    ),
    authors = "Tamás Hakkel",
    format = Documenter.HTML(;
        canonical = "https://hakkelt.github.io/Contourlets.jl",
    ),
    pages = [
        "Home" => "index.md",
        "Guide" => [
            "2-D transforms" => "guide/transforms-2d.md",
            "Denoising and approximation" => "guide/denoising.md",
            "Performance, threading and GPUs" => "guide/performance.md",
        ],
        "Background" => [
            "Theory — 2-D" => "theory.md",
            "Comparison with the MATLAB toolboxes" => "comparison.md",
        ],
        "API Reference" => "api.md",
    ],
    checkdocs = :public,
)

deploydocs(;
    repo = "github.com/hakkelt/Contourlets.jl",
    devbranch = "master",
)
