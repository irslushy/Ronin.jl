push!(LOAD_PATH,"../src/")
using Documenter, RadarQC

makedocs(sitename="RadarQC.jl",
        modules= [RadarQC],
        pages = [
            "Home" => "index.md"
            "Reference" => "api.md"
        ],
        format = Documenter.HTML(prettyurls = false))

deploydocs(;
    repo="github.com/irslushy/RadarQC.jl.git",
    devbranch="main"
)
