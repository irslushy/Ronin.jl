using Pkg

Pkg.instantiate()

using Ronin
using BenchmarkTools
using Missings


###Will quickly run benchmarks on a series of feature calculations for a cfradial 

tasks=["SQI, AHT, STD(VV), PGG, RNG"]

placeholder_matrix = allowmissing(ones(3,3))
weight_matrixes = [allowmissing(ones(7,7)), placeholder_matrix, allowmissing(ones(5,5)),
                    placeholder_matrix, placeholder_matrix, placeholder_matrix]

benchmark_cfrad_locs = "./benchmark_cfrads/"

suite = BenchmarkGroup()

suite["features"] = BenchmarkGroup(["tag1"])
suite["features"][10] = @benchmarkable calculate_features(benchmark_cfrad_locs, tasks, weight_matrixes, 
                                                                "garbage.h5", true; verbose=true, 
                                                                REMOVE_LOW_NCP = true, REMOVE_HIGH_PGG = true, 
                                                                QC_variable="VG", remove_variable="VV")


tune!(suite) 
results = run(suite, verbose=true) 

BenchmarkTools.save("output.json", median(results))

