using Documenter, Contourlets

DocMeta.setdocmeta!(Contourlets, :DocTestSetup, :(using Contourlets, Random); recursive = true)

makedocs(;
    modules = [Contourlets],
    sitename = "Contourlets.jl",
    authors = "Tamás Hakkel",
    format = Documenter.HTML(;
        canonical = "https://hakkelt.github.io/Contourlets.jl",
    ),
    pages = [
        "Home" => "index.md",
        "Guide" => [
            "2-D transforms" => "guide/transforms-2d.md",
            "3-D transforms" => "guide/transforms-3d.md",
            "Denoising and approximation" => "guide/denoising.md",
            "Performance, threading and GPUs" => "guide/performance.md",
        ],
        "Background" => [
            "Theory — 2-D" => "theory.md",
            "Theory — 3-D" => "theory3d.md",
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
