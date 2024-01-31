module RadarQC

    include("./RQCFeatures.jl") 

    using NCDatasets
    using ImageFiltering
    using Statistics
    using Images
    using Missings
    using BenchmarkTools
    using HDF5 
    using MLJ
    using ScikitLearn
    using PyCall, PyCallUtils, BSON

    @sk_import ensemble: RandomForestClassifier
    export get_NCP, airborne_ht, prob_groundgate
    export calc_avg, calc_std, calc_iso, process_single_file 
    export parse_directory, get_num_tasks, get_task_params, remove_validation 

    VARIALBES_TO_QC::Vector{String} = ["ZZ", "VV"]

    function train_model(input_h5::String, model_location::String; verify::Bool=false, verify_out::String="model_verification.h5" )

        ###Load the data
        radar_data = h5open(input_h5)
        printstyled("\nOpening $(radar_data)...\n", color=:blue)
        ###Split into features

        X = read(radar_data["X"])
        Y = read(radar_data["Y"])

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
        BSON.@save model_location currmodel

        if (verify) 
            ###NEW: Write out data to HDF5 files for further processing 
            println("WRITING VERIFICATION DATA TO $(verify_out)" )
            fid = h5open(parsed_args["o"], "w")
            HDF5.write_dataset(fid, "Y_PREDICTED", predicted_Y)
            HDF5.write_dataset(fid, "Y_ACTUAL", Y)
            close(fid) 
        end 

    end     


    ###TODO: Fix arguments etc 
    ###Can have one for a single file and one for a directory 
    function QC_scan(file_path::String, config_file_path::String, indexer_var="VV")

        paths = Vector{String}(file_path)

        for path in paths 
            ##Open in append mode so output variables can be written 
            input_cfrad = NCDataset(path, "a")
            cfrad_dims = (input_cfrad.dim["range"], input_cfrad.dim["time"])

            ###Will generally NOT return Y, but only (X, indexer)
            ###Todo: What do I need to do for parsed args here 
            X, Y, indexer = process_single_file(input_cfrad, config_file_path; REMOVE_HIGH_PGG = true, REMOVE_LOW_NCP = true, remove_variable=indexer_var)

            ##Load saved RF model 
            ##assume that default SYMBOL for saved model is savedmodel
            trained_model = BSON.load(parsed_args["model_path"], @__MODULE__)[:currmodel]
            predictions = ScikitLearn.predict(trained_model, X)

            ##QC each variable in VARIALBES_TO_QC
            for var in VARIALBES_TO_QC

                ##Create new field to reshape QCed field to 
                NEW_FIELD = missings(Float64, cfrad_dims) 

                ##Only modify relevant data based on indexer, everything else should be fill value 
                QCED_FIELDS = input_cfrad[var][:][indexer]
                initial_count = count(!iszero, QCED_FIELDS)
                ##Apply predictions from model 
                ##If model predicts 1, this indicates a prediction of meteorological data 
                QCED_FIELDS = map(x -> Bool(predictions[x[1]]) ? x[2] : JMLQC_utils.FILL_VAL, enumerate(QCED_FIELDS))
                final_count = count(QCED_FIELDS .== JMLQC_utils.FILL_VAL)
                
                ###Need to reconstruct original 
                NEW_FIELD = NEW_FIELD[:]
                NEW_FIELD[indexer] = QCED_FIELDS
                NEW_FIELD = reshape(NEW_FIELD, cfrad_dims)

                defVar(input_cfrad, var * QC_SUFFIX, NEW_FIELD, ("range", "time"), fillvalue = FILL_VAL)

                println()
                printstyled("REMOVED $(initial_count - final_count) PRESUMED NON-METEORLOGICAL DATAPOINTS\n", color=:green)
                println("FINAL COUNT OF DATAPOINTS IN $(var): $(final_count)")

            end 
        end 
    end 


end
