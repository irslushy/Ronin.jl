using NCDatasets
using HDF5
using MLJ
using ScikitLearn
using PyCall, PyCallUtils, BSON
@sk_import ensemble: RandomForestClassifier

###Helper script to apply a trained RF model to a set of mined data parameters 


###Load the data
radar_data = h5open("./ALL_DATA_PROCESSED.h5")

###Split into features

X = read(radar_data["X"])
Y = read(radar_data["Y"])

training_frac = .72
validation_frac = .08 
testing_frac = 1 - training_frac + validation_frac 

###Partition the data accordingly 
println()
println("DATA LOADED SUCESSFULLY. PARTITIONING...")

(training_x, testing_x), (training_y, testing_y) = partition((X, Y), (training_frac + validation_frac), multi=true)
(training_x, validation_x), (training_y, validation_y) = partition((training_x, training_y), training_frac/(training_frac + validation_frac), multi=true)

println("LOADING MODEL: ")
loaded_model = BSON.load("trained_JMLQC_RF.bson", @__MODULE__)[:currmodel]
println("LOADED")
println()

println("TESTING LOADED MODEL") 
println("MODEL VERIFICATION:")
predicted_y = ScikitLearn.predict(loaded_model, testing_x)
accuracy = sum(predicted_y .== testing_y) / length(testing_y)
println("ACCURACY ON TESTING SET: $(round(accuracy * 100, sigdigits=3))%")
println()

fid = h5open("testing_output.h5", "w")
HDF5.write_dataset(fid, "Y_PREDICTED", predicted_y)
HDF5.write_dataset(fid, "Y_ACTUAL", testing_y)
close(fid)

