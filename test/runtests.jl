###Rudimentary test suite to ensure that updates do not break the code 
using Ronin
using Missings 
using HDF5 
using NCDatasets
using BenchmarkTools 
using StatsBase
using Scratch


global scratchspace = @get_scratch!("ronin_testing")


###Will undergo a basic training/QC pipeline. Model is not meant to output 
###Correct results, but rather simply show that it can produce data, train a model, 
###and correctly apply the model to a scan. 

###The below will be testing for a single-pass model

TRAINING_PATH = "../BENCHMARKING/benchmark_cfrads/"
config_file_path = "../BENCHMARKING/benchmark_setup/config.txt"
sample_model = "../BENCHMARKING/benchmark_setup/benchmark_model.joblib"
###Read in tasks

###NEED TO ALLOW THIS TO IGNORE COMMENTS 
tasks = Ronin.get_task_params(config_file_path)

placeholder_matrix = allowmissing(ones(3,3))
center_weight::Float64 = 0

###Weight matrixes for calculating spatial parameters 
iso_weights::Matrix{Union{Missing, Float64}} = allowmissing(ones(7,7))
iso_weights[4,4] = center_weight 
iso_window::Tuple{Int64, Int64} = (7,7)

avg_weights::Matrix{Union{Missing, Float64}} = allowmissing(ones(5,5))
avg_weights[3,3] = center_weight 
avg_window::Tuple{Int64, Int64} = (5,5)

std_weights::Matrix{Union{Missing, Float64}} = allowmissing(ones(5,5))
std_weights[3,3] = center_weight 
std_window::Tuple{Int64, Int64} = (5,5)


weight_matrixes = [placeholder_matrix, placeholder_matrix, std_weights, placeholder_matrix, placeholder_matrix, iso_weights]

path1 = joinpath(scratchspace, "_1.h5")
path2 = joinpath(scratchspace, "_2.h5")

X, Y, idxers_1 = calculate_features(TRAINING_PATH, tasks, weight_matrixes, 
                    path1, true; verbose=true, 
                    REMOVE_LOW_NCP = true, REMOVE_HIGH_PGG = true, 
                    QC_variable="VG", remove_variable="VV", return_idxer = true)


###Ensure multiple dispatch is functioning properly and giving us the same results 
X, Y, idxers_2 = calculate_features(TRAINING_PATH, config_file_path, path2, true,
                    verbose=true, REMOVE_LOW_NCP = true, 
                    REMOVE_HIGH_PGG = true, remove_variable = "VV", QC_variable="VG", return_idxer = true)

@assert idxers_1 == idxers_2 

h5open(path1) do f
    h5open(path2) do f2
        print(f["X"][begin:5,:])
        print(f2["X"][begin:5,:])
        @assert map(round, f["X"][begin:10, :]) == map(round, f2["X"][begin:10, :])
    end
end 

###Also ensure that both are equal to a pre-set version of the benchmark 

print("OK")
h5open(path1) do f 
    h5open("../BENCHMARKING/standard_bench_file.h5") do f2
        @assert map(round, f["X"][:,:]) == map(round, f2["X"][:,:])
    end 
end 

print("OK2")
X = Matrix{Float64}(undef, 0, 6)
Y = Matrix{Float64}(undef, 0, 1)

y_list = []
indexer_list = []

###Get verification for each of the cfradials 
for cfrad_name in readdir(TRAINING_PATH)

    if cfrad_name == ".tmp_hawkedit"
        rm(TRAINING_PATH * cfrad_name, force=true, recursive = true) 
        continue
    end 

    Dataset(TRAINING_PATH * cfrad_name) do f
        X_new, Y_new, indexer = Ronin.process_single_file(f, config_file_path; HAS_INTERACTIVE_QC=true, REMOVE_LOW_NCP=true, REMOVE_HIGH_PGG=true, QC_variable = "VG", 
                                                    remove_variable = "VV", replace_missing = false) 
        global X
        global Y
        X = vcat(X, X_new)
        Y = vcat(Y, Y_new)

        push!(indexer_list, indexer) 
        push!(y_list, Y)
    end 
end 

min_bad_threshold = .9 
min_retain_threshold = .9 

###Now, let's check that we can QC a scan. We'll use the trained ELDORA model here. 
for (i, cfrad_name) in enumerate(readdir(TRAINING_PATH))

    if cfrad_name[1] == '.'
        continue
    end 

    QC_scan(TRAINING_PATH * cfrad_name, config_file_path, sample_model)

    NCDataset(TRAINING_PATH * cfrad_name) do currset
        ###Ensure that the QC is removing at least 90% of bad data and retaining at least 90% of the good data 
        ZZ_QC = currset["ZZ_QC"][:,:][:][indexer_list[i]]
        ZZ_raw = currset["ZZ"][:, :][:][indexer_list[i]]

        ZZ_manual = currset["DBZ"][:, :][:][indexer_list[i]] 
        ###If they are equal, model predicted meteorological 
        model_predictions = [ismissing(x) ? false : true for x in (ZZ_raw .== ZZ_QC)]
        ground_truth = [ismissing(x) ? false : true for x in (ZZ_raw .== ZZ_manual)] 

        true_positives = count(ground_truth[model_predictions .== true] .== true) 
        num_total_positives = count(ground_truth) 

        true_negatives = count(ground_truth[model_predictions .== false] .== false)
        num_total_negatives = count(.!ground_truth) 

        pos_retained_frac = true_positives / num_total_positives
        neg_removed_frac = true_negatives / num_total_negatives
    


        printstyled("Positive Retained Fraction: " * string(pos_retained_frac) * "\n", color = :green) 
        printstyled("Negative Removed Fraction: " * string(neg_removed_frac) * "\n", color=:green)

        @assert pos_retained_frac > min_retain_threshold
        @assert neg_removed_frac > min_bad_threshold

        
        
    end 
end 

###New test in order to test threaded QC_scan feature 
times = []
for i in 1:10
    starttime = time()
    QC_scan(TRAINING_PATH, config_file_path, sample_model)
    push!(times, round(time() - starttime, sigdigits=4))
end 

printstyled("QC_SCAN AVERAGE TIME FOR 10 SCANS: $(mean(times)) seonds \n", color=:green)


###Begin testing the trainin of multi-model configurations 


