###Rudimentary test suite to ensure that updates do not break the code 
using Ronin
using Missings 
using HDF5 
using NCDatasets
using BenchmarkTools 
using StatsBase

###Will undergo a basic training/QC pipeline. Model is not meant to output 
###Correct results, but rather simply show that it can produce data, train a model, 
###and correctly apply the model to a scan. 

TRAINING_PATH = "./BENCHMARKING/benchmark_cfrads/"
config_file_path = "./BENCHMARKING/benchmark_setup/config.txt"

sample_model = "sample_model.jld2"
feature_output = "garbage.h5"
feature_output_2 = "garbage2.h5"

###NEED TO ALLOW THIS TO IGNORE COMMENTS 
tasks = Ronin.get_task_params(config_file_path)

placeholder_matrix = allowmissing(ones(3,3))



center_weight::Float64 = Ronin.center_weight

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

calculate_features(TRAINING_PATH, tasks, weight_matrixes, 
                    feature_output, true; verbose=true, 
                    REMOVE_LOW_NCP = true, REMOVE_HIGH_PGG = true, 
                    QC_variable="VG", remove_variable="VV")


###Ensure multiple dispatch is functioning properly and giving us the same results 
calculate_features(TRAINING_PATH, config_file_path, feature_output_2, true,
                    verbose=true, REMOVE_LOW_NCP = true, 
                    REMOVE_HIGH_PGG = true, remove_variable = "VV", QC_variable="VG")

h5open(feature_output) do f
    h5open(feature_output_2) do f2
        print(f["X"][begin:5,:])
        print(f2["X"][begin:5,:])
        @assert map(round, f["X"][begin:10, :]) == map(round, f2["X"][begin:10, :])
    end
end 

###Also ensure that both are equal to a pre-set version of the benchmark 

h5open(feature_output) do f 
    h5open("./BENCHMARKING/standard_bench_file.h5") do f2
        @assert map(round, f["X"][:,:]) == map(round, f2["X"][:,:])
    end 
end 

X = Matrix{Float64}(undef, 0, 6)
Y = Matrix{Float64}(undef, 0, 1)

y_list = []
indexer_list = []


###train a model 
train_model(feature_output, sample_model)

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

###Test predict_with_model 
_, __ = predict_with_model(sample_model, feature_output)

###Make sure model evaluation and error characteristics at least can be called 
evaluate_model(sample_model, feature_output, config_file_path, mode="H")

###New test in order to test threaded QC_scan feature 
times = []
for i in 1:10
    starttime = time()
    QC_scan(TRAINING_PATH, config_file_path, sample_model, verbose=false)
    push!(times, round(time() - starttime, sigdigits=4))
end 

printstyled("QC_SCAN AVERAGE TIME FOR 10 SCANS: $(mean(times)) seconds \n", color=:green)



horiz_window =  Matrix{Union{Missing, Float64}}(zeros(7,7))
vert_window =  Matrix{Union{Missing, Float64}}(zeros(7,7))

horiz_window[4,:] .= 1.
horiz_window[5,:] .= 1.
horiz_window[3,:] .= 1.

vert_window[:, 4] .= 1.
vert_window[:, 5] .= 1.
vert_window[:, 3] .= 1.
horiz_window_two = horiz_window .* 4

smol_weights = [horiz_window, horiz_window_two]

Dataset(TRAINING_PATH * "/cfrad.19950516_221944.169_to_19950516_221946.124_TF-ELDR_AIR.nc") do currset 
###Ensure that weight matrix version of calculations are consistent between calculate feature versions 
    X, Y, idxer = Ronin.process_single_file(currset, "./BENCHMARKING/micro_tasks.txt", HAS_INTERACTIVE_QC=true, REMOVE_HIGH_PGG=true, REMOVE_LOW_NCP=true, 
                QC_variable="VG", remove_variable="VV", weight_matrixes=smol_weights)

    X2, Y2, idxer2 = Ronin.process_single_file(currset, ["ISO(VV)", "ISO(VV)"], smol_weights, HAS_INTERACTIVE_QC=true, REMOVE_HIGH_PGG=true, REMOVE_LOW_NCP=true, 
                        QC_variable="VG", remove_variable="VV")
    @assert X2 == X
    @assert Y == Y2 
    @assert X[:, 1] != X[:, 2]
end 

printstyled("\n\n...ALL TESTS PASSED...\n\n", color=:green)






