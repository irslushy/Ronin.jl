push!(LOAD_PATH,"../src/")
using Documenter, RadarQC

makedocs(sitename="RadarQC.jl",
        modules= [RadarQC],
        pages = [
            "Home" => "index.md"
        ],
        format = Documenter.HTML(prettyurls = false))

deploydocs(;
    repo="github.com/irslushy/RadarQC.jl.git",
    deploy_config="../.github/workflows/Documentation.yml",
    devbranch="main"
)