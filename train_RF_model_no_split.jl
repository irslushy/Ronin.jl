using NCDatasets
using HDF5
using MLJ
using ScikitLearn
using ArgParse
using PyCall, PyCallUtils, BSON
@sk_import ensemble: RandomForestClassifier


###Set up argument table for CLI 
function parse_commandline()
    
    s = ArgParseSettings()

    @add_arg_table s begin
        
        "training_data"
            help = "Path to input training data h5 file"
            required = true
        "model_output_file"
            help = "Path to output trained/saved RF model to. Recommended to end in .bson"
            required = true
        "-v"
            help = "Whether or not to output verification statistics to a file (true/false)"
            required = false
            default = false 
        "-o"
            help = "Where to output verification data"
            required = false
            default = "trained_model_output.h5"
    end

    return parse_args(s)
end

parsed_args = parse_commandline() 

###Load the data
radar_data = h5open(parsed_args["training_data"])
printstyled("\nOpening $(radar_data)...\n", color=:blue)
###Split into features

X = read(radar_data["X"])
Y = read(radar_data["Y"])

println("TRAINING ON $(size(X)[1]) RADAR GATES\n")

###Determine the class weights 
MET_LENGTH = length(Y[Y[:] .== 1])
MET_FRAC = (MET_LENGTH) / length(Y) 
NON_MET_FRAC = (length(Y) - MET_LENGTH) / length(Y)

println("CLASS BALANCE: $(round(MET_FRAC * 100, sigdigits = 3))% METEOROLOGICAL DATA, $(round(NON_MET_FRAC * 100, sigdigits = 3))% NON METEOROLOGICAL DATA")
println()


###Taken from sklearn documentation 
###This is a little bit overkill, but a safer way to do this that is more extendible to greater class problems 
currmodel = RandomForestClassifier(n_estimators = 21, max_depth = 14, random_state = 50, class_weight = "balanced")
#println(typeof(currmodel))
println("FITTING MODEL")
startTime = time() 
ScikitLearn.fit!(currmodel, X, reshape(Y, length(Y),))
println("COMPLETED FITTING MODEL IN $((time() - startTime)) seconds")
println() 

currmodel = BSON.load("trained_JMLQC_RF.bson", @__MODULE__)[:currmodel]

println("MODEL VERIFICATION:")
predicted_Y = ScikitLearn.predict(currmodel, X)
accuracy = sum(predicted_Y .== Y) / length(Y)
println("ACCURACY ON TRAINING SET: $(round(accuracy * 100, sigdigits=3))%")
println()


println("SAVING MODEL: ") 
BSON.@save parsed_args["model_output_file"] currmodel
parsebool() = lowercase(parsed_args["v"]) == "true" 

if (parsebool()) 
    ###NEW: Write out data to HDF5 files for further processing 
    fid = h5open(parsed_args["o"], "w")
    HDF5.write_dataset(fid, "Y_PREDICTED", predicted_Y)
    HDF5.write_dataset(fid, "Y_ACTUAL", Y)
    close(fid) 
end 
