using Documenter, Contourlets

DocMeta.setdocmeta!(Contourlets, :DocTestSetup, :(using Contourlets, Random); recursive = true)

makedocs(;
    modules = [Contourlets],
    sitename = "Contourlets.jl",
    authors = "Contourlets.jl Contributors",
    repo = "https://github.com/hakkelt/Contourlets.jl/blob/{commit}{path}#{line}",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://hakkelt.github.io/Contourlets.jl",
        edit_link = "master",
        assets = String[],
    ),
    pages = [
        "Home" => "index.md",
        "Theory" => "theory.md",
        "API Reference" => "api.md",
        "Examples" => [
            "Showcase" => "examples/showcase.md",
            "Contourlet Transform" => "examples/ct_example.md",
            "Nonsubsampled Contourlet Transform" => "examples/nsct_example.md",
            "Approximation & Denoising" => "examples/nla_denoising.md",
        ],
        "Comparison with MATLAB and Python" => "comparison.md",
    ],
    checkdocs = :exports,
    doctest = true,
)

deploydocs(;
    repo = "github.com/hakkelt/Contourlets.jl",
    devbranch = "master",
)
