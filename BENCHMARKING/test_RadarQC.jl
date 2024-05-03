###Rudimentary test suite to ensure that updates do not break the code 
using RadarQC
using Missings 
using HDF5 

###Will undergo a basic training/QC pipeline. Model is not meant to output 
###Correct results, but rather simply show that it can produce data, train a model, 
###and correctly apply the model to a scan. 

TRAINING_PATH = "./benchmark_cfrads"
config_file = "./sample_tasks.txt"


###Read in tasks

tasks = open(config_file) do file
    map(String, split(filter(s -> !isspace(s), read(file, String)), ","))
end 
print(tasks)

placeholder_matrix = allowmissing(ones(3,3))


center_weight = 0

std_weights = allowmissing(ones(5,5))
std_weights[3,3] = center_weight 

weight_matrixes = [allowmissing(ones(7,7)), placeholder_matrix, std_weights,
                    placeholder_matrix, placeholder_matrix, placeholder_matrix]

calculate_features(TRAINING_PATH, tasks, weight_matrixes, 
                    "garbage.h5", true; verbose=true, 
                    REMOVE_LOW_NCP = true, REMOVE_HIGH_PGG = true, 
                    QC_variable="VG", remove_variable="VV")

###Ensure multiple dispatch is functioning properly and giving us the same results 
calculate_features(TRAINING_PATH, config_file, "garbage_2.h5", true,
                    verbose=true, REMOVE_LOW_NCP = true, 
                    REMOVE_HIGH_PGG = true, remove_variable = "VV", QC_variable="VG")

h5open("garbage.h5") do f
    h5open("garbage_2.h5") do f2
        @assert f["X"][:, :] == f2["X"][:, :]
    end
end 


###Now, let's check that we can QC a scan. We'll use the NOAA model here. 