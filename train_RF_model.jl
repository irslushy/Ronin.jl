using NCDatasets
using HDF5
using MLJ
using ScikitLearn
using PyCall, BSON

@sk_import ensemble: RandomForestClassifier

###Load the data
radar_data = h5open("./ALEX_VERIFICATION/mined_out.h5")

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

if round(size(training_x)[1] / size(X)[1], sigdigits = 2) != training_frac
    println("ERROR: EXPECTED TRAINING FRACION OF $training_frac")
    println("GOT TRAINING FRACTION OF $(round(size(training_x)[1], sigdigits = 2))")
end

###Determine the class weights 
MET_LENGTH = length(Y[Y[:] .== 1])
MET_FRAC = (MET_LENGTH) / length(Y) 
NON_MET_FRAC = (length(Y) - MET_LENGTH) / length(Y)

println("CLASS BALANCE: $(round(MET_FRAC * 100, sigdigits = 3))% METEOROLOGICAL DATA, $(round(NON_MET_FRAC * 100, sigdigits = 3))% NON METEOROLOGICAL DATA")
println()

###Taken from sklearn documentation 
###This is a little bit overkill, but a safer way to do this that is more extendible to greater class problems 
currmodel = RandomForestClassifier(n_estimators = 21, max_depth = 14, n_jobs = -1, random_state = 50, class_weight = "balanced")

println("FITTING MODEL")
startTime = time() 
ScikitLearn.fit!(currmodel, training_x, reshape(training_y, length(training_y),))
println("COMPLETED FITTING MODEL IN $((time() - startTime)) seconds")
println() 

println("MODEL VERIFICATION:")
predicted_Y = ScikitLearn.predict(currmodel, testing_x)
accuracy = sum(predicted_Y .== testing_y) / length(testing_y)
println("ACCURACY ON TESTING SET: $(round(accuracy * 100, sigdigits=3))%")
println()

println("SAVING MODEL: ") 
bson("trained_JMLQC_RF.bson", Dict(:a => currmodel))

println("LOADING MODEL: ")
loaded_model = BSON.load("trained_JMLQC_RF.bson", @__MODULE__)[:a]
println("LOADED")
println()

println("TESTING LOADED MODEL") 
println("MODEL VERIFICATION:")
predicted_Y = ScikitLearn.predict(currmodel, testing_x)
accuracy = sum(predicted_Y .== testing_y) / length(testing_y)
println("ACCURACY ON TESTING SET: $(round(accuracy * 100, sigdigits=3))%")
println()
###@TODO cut out the predictors eliminated by lasso regression

##Initialize the RF model to the specified hyperparameters
##@TODO figure out how to implement balanced class weights 

##Train the model using the combined arrays 

###Evaluation metrics to look at:  
    # -Confusion matrix
    # -Normalized confusion matrix 
    # -Accuracy 
    # -Weighted F1 Score 
    #ROC-AUC score 

