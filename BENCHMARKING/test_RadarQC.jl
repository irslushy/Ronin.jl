###Rudimentary test suite to ensure that updates do not break the code 
using RadarQC
using Missings 
using HDF5 
using NCDatasets

###Will undergo a basic training/QC pipeline. Model is not meant to output 
###Correct results, but rather simply show that it can produce data, train a model, 
###and correctly apply the model to a scan. 

TRAINING_PATH = "./BENCHMARKING/benchmark_cfrads/"
config_file_path = "./BENCHMARKING/benchmark_setup/config.txt"
sample_model = "./BENCHMARKING/benchmark_setup/benchmark_model.joblib"
###Read in tasks


###NEED TO ALLOW THIS TO IGNORE COMMENTS 
tasks = RadarQC.get_task_params(config_file_path)
print(tasks)

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

calculate_features(TRAINING_PATH, tasks, weight_matrixes, 
                    "garbage.h5", true; verbose=true, 
                    REMOVE_LOW_NCP = true, REMOVE_HIGH_PGG = true, 
                    QC_variable="VG", remove_variable="VV")

###Ensure multiple dispatch is functioning properly and giving us the same results 
calculate_features(TRAINING_PATH, config_file_path, "garbage_2.h5", true,
                    verbose=true, REMOVE_LOW_NCP = true, 
                    REMOVE_HIGH_PGG = true, remove_variable = "VV", QC_variable="VG")

h5open("garbage.h5") do f
    h5open("garbage_2.h5") do f2
        print(count(f["X"][:, :] .== f2["X"][:,:]))
        print(length(f["X"][:,:][:]))
        @assert f["X"][:, :] == f2["X"][:, :]
    end
end 


X = Matrix{Float64}(undef, 0, 6)
Y = Matrix{Float64}(undef, 0, 1)

y_list = []
indexer_list = []

###Get verification for each of the cfradials 
for cfrad_name in readdir(TRAINING_PATH)

    Dataset(TRAINING_PATH * cfrad_name) do f
        X_new, Y_new, indexer = RadarQC.process_single_file(f, config_file_path; HAS_MANUAL_QC=true, REMOVE_LOW_NCP=true, REMOVE_HIGH_PGG=true, QC_variable = "VG", 
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
        
        
        @assert pos_retained_frac > min_retain_threshold
        @assert neg_removed_frac > min_bad_threshold

        printstyled("Positive Retained Fraction: " * string(pos_retained_frac), color = :green) 
        printstyled("Negative Removed Fraction: " * string(neg_removed_frac), color=:green)
        
    end 
    
end 








