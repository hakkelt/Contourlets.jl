using Documenter, Contourlets, MultiscaleFilterBanks, AbstractMultiscaleTransforms

# Doctests run in every spliced docstring, including those of the two dependencies.
DocMeta.setdocmeta!(Contourlets, :DocTestSetup, :(using Contourlets, Random); recursive = true)
DocMeta.setdocmeta!(
    MultiscaleFilterBanks, :DocTestSetup,
    :(using MultiscaleFilterBanks, Random); recursive = true
)
DocMeta.setdocmeta!(
    AbstractMultiscaleTransforms, :DocTestSetup,
    :(using AbstractMultiscaleTransforms); recursive = true
)

makedocs(;
    # Documenter finds docstrings only in the listed modules.  The two
    # dependencies are listed for the names Contourlets re-exports (`FilterPair`,
    # `CDF97`, the interface); their other public names are documented in their
    # own manuals, so `checkdocs` skips them.
    modules = [Contourlets, MultiscaleFilterBanks, AbstractMultiscaleTransforms],
    checkdocs_ignored_modules = [MultiscaleFilterBanks, AbstractMultiscaleTransforms],
    sitename = "Contourlets.jl",
    # Docstrings spliced from the two dependencies link to their repositories.
    remotes = Dict(
        pkgdir(MultiscaleFilterBanks) =>
            Documenter.Remotes.GitHub("hakkelt", "MultiscaleFilterBanks.jl"),
        pkgdir(AbstractMultiscaleTransforms) =>
            Documenter.Remotes.GitHub("hakkelt", "AbstractMultiscaleTransforms.jl"),
    ),
    authors = "Tamás Hakkel",
    format = Documenter.HTML(;
        canonical = "https://hakkelt.github.io/Contourlets.jl",
    ),
    pages = [
        "Home" => "index.md",
        "How-to guides" => [
            "Choose a transform" => "guide/transforms.md",
            "Reuse a workspace" => "guide/workspaces.md",
            "Denoise with the NSCT" => "guide/denoising.md",
            "Threads and GPUs" => "guide/parallel.md",
            "Pyramid and filter-bank stages" => "guide/stages.md",
        ],
        "Explanation" => ["Theory" => "theory.md"],
        "Reference" => ["API" => "api.md"],
    ],
    checkdocs = :public,
)

deploydocs(;
    repo = "github.com/hakkelt/Contourlets.jl",
    devbranch = "master",
)
