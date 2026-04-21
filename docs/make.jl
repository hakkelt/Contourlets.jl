using Documenter, Contourlets

DocMeta.setdocmeta!(Contourlets, :DocTestSetup, :(using Contourlets, Random); recursive = true)

makedocs(;
    modules = [Contourlets],
    sitename = "Contourlets.jl",
    authors = "Contourlets.jl Contributors",
    repo = "https://github.com/your-org/Contourlets.jl/blob/{commit}{path}#{line}",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://your-org.github.io/Contourlets.jl",
        edit_link = "master",
        assets = String[],
    ),
    pages = [
        "Home" => "index.md",
        "Theory" => "theory.md",
        "API Reference" => "api.md",
        "Examples" => [
            "Contourlet Transform" => "examples/ct_example.md",
            "Nonsubsampled Contourlet Transform" => "examples/nsct_example.md",
        ],
    ],
    checkdocs = :exports,
    doctest = true,
    warnonly = [:doctest, :missing_docs],
)

deploydocs(;
    repo = "github.com/your-org/Contourlets.jl",
    devbranch = "master",
)
