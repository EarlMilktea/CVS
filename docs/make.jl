using CVS
using Documenter

DocMeta.setdocmeta!(CVS, :DocTestSetup, :(using CVS); recursive = true)

makedocs(;
    modules = [CVS],
    authors = "SS",
    repo = "https://github.com/EarlMilktea/CVS.jl/blob/{commit}{path}#{line}",
    sitename = "CVS.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://EarlMilktea.github.io/CVS.jl",
        edit_link = "main",
        assets = String[],
    ),
    pages = ["Home" => "index.md"],
)

deploydocs(; repo = "github.com/EarlMilktea/CVS.jl", devbranch = "main")
