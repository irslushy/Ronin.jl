# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .jl
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Julia 1.10.2
#     language: julia
#     name: julia-1.10
# ---

# <h1>Start by activating our project enviornment and installing required dependencies</h1>

# +
using Pkg
Pkg.activate(".")
Pkg.instantiate() 
##Make sure Julia can see our module 
push!(LOAD_PATH, "./src/")

###Load key functionality 
###This will take a while the first time you do it 
using RadarQC
# -

# <h2>Let's begin by splitting our CFRadials into a Training and a Testing directory</h2>

# +
###Make sure to use absolute paths here 
DATA_PATH = "/path/to/RadarQC_training"
CASE_PATHS= [string(DATA_PATH,"/BAMEX"),
             string(DATA_PATH,"/HAGUPIT"), 
             string(DATA_PATH,"/RITA"), 
             string(DATA_PATH,"/VORTEX")]


#Create Training and Testing Directories
PWD = pwd()

TRAINING_PATH = string(PWD,"/TRAINING")
TESTING_PATH = string(PWD,"/TESTING")

if isdir(TRAINING_PATH)
    print("Training Directory exists\n")
else
    mkdir(TRAINING_PATH)
end

if isdir(TESTING_PATH)
    print("Testing Directory exists\n")
else
    mkdir(TESTING_PATH)
end
# -

@time split_training_testing!(CASE_PATHS, TRAINING_PATH, TESTING_PATH)

# <h2>Now, calculate input features based off of the training/testing data<h2> 
# <h1><span style="color:Green">This takes up a lot of memory in a notebook, so I would recommend running this in a script</span></h1>
#
#
#

# +
###Arguments are
###(input) Path to training file/directory 
###(input) Path to file containing arguments on features to calculate 
###(Output) Name of output file containing calculated features 

###This will use the 5-parameter model described in the referenced manuscript 

###You can specified the verbose=false flag so as to not overload notebook memory
###Generally diagnostics about timing will print out 


@time calculate_features(TRAINING_PATH, "./MODELS/DesRosiers_Bell_23/config.txt", "training_features.h5", true,
                    verbose=true, REMOVE_LOW_NCP = true, 
                    REMOVE_HIGH_PGG = true, remove_variable = "VV" )
# -

@time calculate_features(TESTING_PATH, "./MODELS/DesRosiers_Bell_23/config.txt", "testing_features.h5", true,
                    verbose=false, REMOVE_LOW_NCP = true, 
                    REMOVE_HIGH_PGG = true, remove_variable = "VV")

###Remove validation 
###keyword args 
###training_output -- path to output training set to 
###validation_output -- path to output validation set to 
@time remove_validation("training_features.h5")

# ## Now - train the model! This could take on the order of 20-30 minutes if training on something the size of the ELDORA dataset

# +
###Now, let's train our model on it! 
@time RadarQC.train_model("training_features.h5", "TRAINED_ELDORA.joblib")

#@time RadarQC.train_model("training_features.h5", "TRAINED_ELDORA.joblib";
#             verify=true, verify_out="NOAA_MODEL_verification.h5")
# -

# <h1>Now, select variables you wish to QC and apply the model to them!</h1> 

###These need to have the same name as in the cfradial file
VARS_TO_QC = ["ZZ", "VV"]
###This will write out the new QC'ed variables to the same cfrad file with the name ZZ_QC and VV_QC 
@time QC_scan("./BENCHMARKING/benchmark_cfrads/cfrad.19950516_221950.411_to_19950516_221953.219_TA-ELDR_AIR.nc", 
        "./MODELS/DesRosiers_Bell_23/config.txt", 
        "TRAINED_ELDORA.joblib"; 
        VARIABLES_TO_QC = VARS_TO_QC, 
        QC_suffix="_QC")
