push!(LOAD_PATH,"../src/")
using Documenter, Ronin

makedocs(sitename="Ronin.jl",
        modules= [Ronin],
        pages = [
            "Home" => "index.md"
            "Reference" => "api.md"
        ],
        format = Documenter.HTML(prettyurls = false), 
        checkdocs=:none)

deploydocs(;
    repo="github.com/irslushy/Ronin.jl.git",
    devbranch="main"
)
