module RadarQC

    include("./RQCFeatures.jl") 
    include("./Io.jl")

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
    export calculate_features
    export split_training_testing 

    VARIALBES_TO_QC::Vector{String} = ["ZZ", "VV"]
    QC_SUFFIX::String = "_QC"


    """
    Function to process a set of cfradial files and produce a set of input features for training/evaluating a model 
        input_loc: cfradial files are specified by input_loc - can be either a file or a directory
        arg_file: Features to be calculated are specified in a file located at arg_file 
        output_file : Location to output the resulting dataset
    """
    function calculate_features(input_loc::String, argument_file::String, output_file::String)

        ##If this is a directory, things get a little more complicated 
        paths = Vector{String}()
    
        if isdir(input_loc) 
            paths = parse_directory(input_loc)
        else 
            paths = [input_loc]
        end 
        
        ###Setup h5 file for outputting mined parameters
        ###processing will proceed in order of the tasks, so 
        ###add these as an attribute akin to column headers in the H5 dataset
        ###Also specify the fill value used 
    
        println("OUTPUTTING DATA IN HDF5 FORMAT TO FILE: $(output_file)")
        fid = h5open(output_file, "w")
    
        ###Add information to output h5 file 
        #attributes(fid)["Parameters"] = tasks
        attributes(fid)["MISSING_FILL_VALUE"] = FILL_VAL
    
        ##Instantiate Matrixes to hold calculated features and verification data 
        output_cols = get_num_tasks(argument_file)
    
        newX = X = Matrix{Float64}(undef,0,output_cols)
        newY = Y = Matrix{Int64}(undef, 0,1) 
    
        
        starttime = time() 
    
        for (i, path) in enumerate(paths) 
            try 
                cfrad = Dataset(path) 
                (newX, newY, indexer) = process_single_file(cfrad, argument_file)
            catch e
                if isa(e, DimensionMismatch)
                    printstyled(Base.stderr, "POSSIBLE ERRONEOUS CFRAD DIMENSIONS... SKIPPING $(path)\n"; color=:red)
                else 
                    printstyled(Base.stderr, "UNRECOVERABLE ERROR\n"; color=:red)
                    throw(e)
    
                ##@TODO CATCH exception handling for invalid task 
                end
            else
                X = vcat(X, newX)::Matrix{Float64}
                Y = vcat(Y, newY)::Matrix{Int64}
            end
        end 
    
        println("COMPLETED PROCESSING $(length(paths)) FILES IN $(round((time() - starttime), digits = 0)) SECONDS")
    
        ###Get verification information 
        ###0 indicates NON METEOROLOGICAL data that was removed during manual QC
        ###1 indicates METEOROLOGICAL data that was retained during manual QC 
        
        ##Probably only want to write once, I/O is very slow 
        println()
        println("WRITING DATA TO FILE OF SHAPE $(size(X))")
        X
        println("X TYPE: $(typeof(X))")
        write_dataset(fid, "X", X)
        write_dataset(fid, "Y", Y)
        close(fid)

    end 


    function train_model(input_h5::String, model_location::String; verify::Bool=false, verify_out::String="model_verification.h5" )

        ###Load the data
        radar_data = h5open(input_h5)
        printstyled("\nOpening $(radar_data)...\n", color=:blue)
        ###Split into features

        X = read(radar_data["X"])
        Y = read(radar_data["Y"])

        currmodel= RandomForestClassifier(n_estimators = 21, max_depth = 14, random_state = 50, class_weight = "balanced")

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
    """
    Primary function to apply a trained RF model to a cfradial scan 
    """
    function QC_scan(file_path::String, config_file_path::String, model_path::String; indexer_var="VV")

        paths = Vector{String}()
        push!(paths, file_path)

        for path in paths 
            ##Open in append mode so output variables can be written 
            input_cfrad = NCDataset(path, "a")
            cfrad_dims = (input_cfrad.dim["range"], input_cfrad.dim["time"])

            ###Will generally NOT return Y, but only (X, indexer)
            ###Todo: What do I need to do for parsed args here 
            X, Y, indexer = process_single_file(input_cfrad, config_file_path; REMOVE_HIGH_PGG = true, REMOVE_LOW_NCP = true, remove_variable=indexer_var)

            ##Load saved RF model 
            ##assume that default SYMBOL for saved model is savedmodel
            trained_model = BSON.load(model_path, @__MODULE__)[:currmodel]
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
                QCED_FIELDS = map(x -> Bool(predictions[x[1]]) ? x[2] : FILL_VAL, enumerate(QCED_FIELDS))
                final_count = count(QCED_FIELDS .== FILL_VAL)
                
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
