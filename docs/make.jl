using Lorenz96andNDE
using Documenter

DocMeta.setdocmeta!(Lorenz96andNDE, :DocTestSetup, :(using Lorenz96andNDE); recursive=true)

makedocs(;
    modules=[Lorenz96andNDE],
    authors="Lisa Kauck <lisa.kauck@tum.de",
    repo="https://github.com/LisaMarieKauck/Lorenz96andNDE.jl/blob/{commit}{path}#{line}",
    sitename="Lorenz96andNDE.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        edit_link="master",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)
