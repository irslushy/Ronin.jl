###Function containing capabilites to split input datasets into different training/testing compoments and Otherwise
###Input/output functionality 
export remove_validation 
function remove_validation(input_dataset::String; training_output="train_no_validation_set.h5", validation_output = "validation.h5")
        
    currset = h5open(input_dataset)
    params = attrs(currset)["Parameters"]
    FILL_VAL = attrs(currset)["MISSING_FILL_VALUE"]

    X = currset["X"][:,:]
    Y = currset["Y"][:,:] 

    ###Will include every STEP'th feature in the validation dataset 
    ###A step of 10 will mean that every 10th will feature in the validation set, and everything else in training 
    STEP = 10 

    ###Make every STEPth index false in the testing indexer, and true in the validation indexer 
    test_indexer = [true for i=1:size(X)[1]]
    test_indexer[begin:STEP:end] .= false

    validation_indexer = .!test_indexer 

    validation_out = h5open(validation_output, "w")
    training_out = h5open(training_output, "w")
    
    write_dataset(validation_out, "X", X[validation_indexer, :])
    write_dataset(validation_out, "Y", Y[validation_indexer, :])
    attributes(validation_out)["MISSING_FILL_VALUE"] = FILL_VAL
    attributes(validation_out)["Parameters"] = params


    write_dataset(training_out, "X", X[test_indexer, :])
    write_dataset(training_out, "Y", Y[test_indexer, :])
    attributes(training_out)["MISSING_FILL_VALUE"] = FILL_VAL
    attributes(training_out)["Parameters"] = params

    close(currset)
    close(validation_out)
    close(training_out)

end


