using Documenter, Contourlets, MultiscaleFilterBanks

DocMeta.setdocmeta!(Contourlets, :DocTestSetup, :(using Contourlets, Random); recursive = true)
DocMeta.setdocmeta!(
    MultiscaleFilterBanks, :DocTestSetup,
    :(using MultiscaleFilterBanks, Random); recursive = true
)

makedocs(;
    # MultiscaleFilterBanks is in the list because Contourlets re-exports its
    # `FilterPair` and `CDF97`, whose docstrings live there.  Its other public
    # names are documented in its own manual, so `checkdocs` skips it.
    modules = [Contourlets, MultiscaleFilterBanks],
    checkdocs_ignored_modules = [MultiscaleFilterBanks],
    sitename = "Contourlets.jl",
    remotes = Dict(
        pkgdir(MultiscaleFilterBanks) =>
            Documenter.Remotes.GitHub("hakkelt", "MultiscaleFilterBanks.jl"),
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
