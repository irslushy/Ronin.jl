using NCDatasets
using HDF5
using MLJ


###Load the data
radar_data = h5open("./ALEX_VERIFICATION/mined_out.h5")

###Split into features

X = read(radar_data["X"])
Y = read(radar_data["Y"])

training_frac = .72
validation_frac = .08 
testing_frac = 1 - training_frac + validation_frac 

###Partition the data accordingly 
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

println("CLASS BALANCE: $(MET_FRAC) % METEOROLOGICAL DATA, $(NON_MET_FRAC)% NON METEOROLOGICAL DATA")
###Class weight formula wj = n_samples/(n_classes*n_samples)
###Taken from sklearn documentation 
###This is a little bit overkill, but a safer way to do this that is more extendible to greater class problems 
classes = Set(Y)
class_weights = Dict{Int64, Float64}()
for class in classes
    println("CLASS: $(class)")
end 


###@TODO cut out the predictors eliminated by lasso regression

##Initialize the RF model to the specified hyperparameters
##@TODO figure out how to implement balanced class weights 

##Train the model using the combined arrays 

###Evaluation metrics to look at:  
    # -Confusion matrixl
    # -Normalized confusion matrix 
    # -Accuracy 
    # -Weighted F1 Score 
    #ROC-AUC score 

