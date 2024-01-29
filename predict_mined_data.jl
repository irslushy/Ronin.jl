using NCDatasets
using HDF5
using MLJ
using ScikitLearn
using PyCall, PyCallUtils, BSON
using ArgParse 
@sk_import ensemble: RandomForestClassifier

###Helper script to apply a trained RF model to a set of mined data parameters 

###Set up argument table for CLI 
function parse_commandline()
    
    s = ArgParseSettings()

    @add_arg_table s begin
        
        "input_data"
            help = "Path to input data h5 file"
            required = true
        "model_location"
            help = "Path to trained/saved RF model . Recommended to end in .bson"
            required = true
        "output_location"
            help = "Location to output h5 with predicted and actual data to"
            required = true
    end

    return parse_args(s)
end

parsed_args = parse_commandline()

###Load the data
radar_data = h5open(parsed_args["input_data"])
###Split into features

X = read(radar_data["X"])
Y = read(radar_data["Y"])

println("LOADING MODEL: ")
loaded_model = BSON.load(parsed_args["model_location"], @__MODULE__)[:currmodel]
println("LOADED")
println()

println("TESTING LOADED MODEL") 
println("MODEL VERIFICATION:")
predicted_y = ScikitLearn.predict(loaded_model, X)
accuracy = sum(predicted_y .== Y) / length(Y)
println("ACCURACY ON TESTING SET: $(round(accuracy * 100, sigdigits=3))%")
println()

fid = h5open(parsed_args["output_location"], "w")
HDF5.write_dataset(fid, "Y_PREDICTED", predicted_y)
HDF5.write_dataset(fid, "Y_ACTUAL", Y)
close(fid)

