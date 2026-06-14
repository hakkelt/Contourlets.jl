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
        "Theory" => "theory.md",
        "Examples" => [
            "Showcase" => "examples/showcase.md",
            "Contourlet Transform" => "examples/ct_example.md",
            "Nonsubsampled Contourlet Transform" => "examples/nsct_example.md",
            "Approximation & Denoising" => "examples/nla_denoising.md",
        ],
        "Comparison with MATLAB" => "comparison.md",
        "API Reference" => "api.md",
    ],
    checkdocs = :exports,
)

deploydocs(;
    repo = "github.com/hakkelt/Contourlets.jl",
    devbranch = "master",
)
