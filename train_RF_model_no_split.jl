using NCDatasets
using HDF5
using MLJ
using ScikitLearn
using PyCall, PyCallUtils, BSON
@sk_import ensemble: RandomForestClassifier


###Load the data
radar_data = h5open("TRAINING_DATA_NO_VALID.h5")
printstyled("\nOpening $(radar_data)...\n", color=:blue)
sleep(2)
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
println(typeof(currmodel))
println("FITTING MODEL")
startTime = time() 
ScikitLearn.fit!(currmodel, X, reshape(Y, length(Y),))
println("COMPLETED FITTING MODEL IN $((time() - startTime)) seconds")
println() 


println("MODEL VERIFICATION:")
predicted_Y = ScikitLearn.predict(currmodel, X)
accuracy = sum(predicted_Y .== Y) / length(Y)
println("ACCURACY ON TRAINING SET: $(round(accuracy * 100, sigdigits=3))%")
println()


println("SAVING MODEL: ") 
BSON.@save "trained_JMLQC_RF.bson" currmodel


###NEW: Write out data to HDF5 files for further processing 
fid = h5open("testing_output.h5", "w")
HDF5.write_dataset(fid, "Y_PREDICTED", predicted_Y)
HDF5.write_dataset(fid, "Y_ACTUAL", Y)
close(fid) 