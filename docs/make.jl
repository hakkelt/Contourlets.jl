using Documenter, Contourlets, MultiscaleFilterBanks

DocMeta.setdocmeta!(Contourlets, :DocTestSetup, :(using Contourlets, Random); recursive = true)
DocMeta.setdocmeta!(
    MultiscaleFilterBanks, :DocTestSetup,
    :(using MultiscaleFilterBanks, Random); recursive = true
)

makedocs(;
    # MultiscaleFilterBanks is in the list because Contourlets re-exports its
    # public names (`FilterPair`, `CDF97`, the threading policies), and their
    # docstrings live there.
    modules = [Contourlets, MultiscaleFilterBanks],
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
    checkdocs = :exports,
)

deploydocs(;
    repo = "github.com/hakkelt/Contourlets.jl",
    devbranch = "master",
)
