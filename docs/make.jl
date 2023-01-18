using CVS
using Documenter

DocMeta.setdocmeta!(CVS, :DocTestSetup, :(using CVS); recursive = true)

makedocs(;
    modules = [CVS],
    authors = "SS",
    repo = "https://github.com/EarlMilktea/CVS/blob/{commit}{path}#{line}",
    sitename = "CVS",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://EarlMilktea.github.io/CVS",
        edit_link = "main",
        assets = String[],
    ),
    pages = ["Home" => "index.md"],
)

deploydocs(; repo = "github.com/EarlMilktea/CVS", devbranch = "main")
