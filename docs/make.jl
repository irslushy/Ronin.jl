push!(LOAD_PATH,"../src/")
using Documenter, RadarQC

makedocs(sitename="RadarQC.jl",
        modules= [RadarQC],
        pages = [
            "Home" => "index.md"
        ])

deploydocs(;
    repo="github.com/irslushy/RadarQC.jl",
    devbranch="main"
)