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
    using PythonCall

    export get_NCP, airborne_ht, prob_groundgate
    export calc_avg, calc_std, calc_iso, process_single_file 
    export parse_directory, get_num_tasks, get_task_params, remove_validation 
    export calculate_features
    export split_training_testing! 
    export train_model 
    export QC_scan 

    """
    Function to process a set of cfradial files and produce a set of input features for training/evaluating a model 
        input_loc: cfradial files are specified by input_loc - can be either a file or a directory
        argumet_file: Features to be calculated are specified in a file located at arg_file 
        output_file : Location to output the resulting dataset

        Keyword Arguments: 
        remove_variable - variable in CFRadial file used to determine where 'missing' gates are. 
                          these gates will be removed from the outputted features so that the model 
                          is not trained on missing data
    """
    function calculate_features(input_loc::String, argument_file::String, output_file::String; verbose=false,
        HAS_MANUAL_QC = false, REMOVE_LOW_NCP = false, REMOVE_HIGH_PGG = false, QC_variable = "VG", remove_variable = "VV")

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
        attributes(fid)["Parameters"] = get_task_params(argument_file)
        attributes(fid)["MISSING_FILL_VALUE"] = FILL_VAL
    
        ##Instantiate Matrixes to hold calculated features and verification data 
        output_cols = get_num_tasks(argument_file)
    
        newX = X = Matrix{Float64}(undef,0,output_cols)
        newY = Y = Matrix{Int64}(undef, 0,1) 
    
        
        starttime = time() 
    
        for (i, path) in enumerate(paths) 
            try 
                cfrad = Dataset(path) 
                pathstarttime=time() 
                (newX, newY, indexer) = process_single_file(cfrad, argument_file; 
                                            HAS_MANUAL_QC = HAS_MANUAL_QC, REMOVE_LOW_NCP = REMOVE_LOW_NCP, 
                                            REMOVE_HIGH_PGG = REMOVE_HIGH_PGG, QC_variable = QC_variable, remove_variable = remove_variable)
                close(cfrad)

                if verbose
                    println("Processed $(path) in $(time()-pathstarttime) seconds")
                end 

            catch e
                if isa(e, DimensionMismatch)
                    printstyled(Base.stderr, "POSSIBLE ERRONEOUS CFRAD DIMENSIONS... SKIPPING $(path)\n"; color=:red)
                else 
                    printstyled(Base.stderr, "UNRECOVERABLE ERROR\n"; color=:red)
                    close(fid)
                    throw(e)
    
                ##@TODO CATCH exception handling for invalid task 
                end
            else
                X = vcat(X, newX)::Matrix{Float64}
                Y = vcat(Y, newY)::Matrix{Int64}
            end
        end 
    
        println("COMPLETED PROCESSING $(length(paths)) FILES IN $(round((time() - starttime), digits = 2)) SECONDS")
    
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


        ###Import necessecary Python modules 
        joblib = pyimport("joblib")
        ensemble = pyimport("sklearn.ensemble")

        ###Load the data
        radar_data = h5open(input_h5)
        printstyled("\nOpening $(radar_data)...\n", color=:blue)
        ###Split into features

        X = read(radar_data["X"])
        Y = read(radar_data["Y"])

        model = ensemble.RandomForestClassifier(n_estimators = 21, max_depth = 14, random_state = 50, class_weight = "balanced")

        println("FITTING MODEL")
        startTime = time() 
        model.fit(X, reshape(Y, length(Y),))
        println("COMPLETED FITTING MODEL IN $((time() - startTime)) seconds")
        println() 


        println("MODEL VERIFICATION:")
        predicted_Y = pyconvert(Vector{Float64}, model.predict(X))
        accuracy = sum(predicted_Y .== Y) / length(Y)
        println("ACCURACY ON TRAINING SET: $(round(accuracy * 100, sigdigits=3))%")
        println()


        println("SAVING MODEL: ") 
        joblib.dump(model, model_location)

        if (verify) 
            ###NEW: Write out data to HDF5 files for further processing 
            println("WRITING VERIFICATION DATA TO $(verify_out)" )
            fid = h5open(verify_out, "w")
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
    function QC_scan(file_path::String, config_file_path::String, model_path::String; VARIABLES_TO_QC = ["ZZ", "VV"], QC_suffix = "_QC", indexer_var="VV")
        
        joblib = pyimport("joblib") 

        paths = Vector{String}() 
        if isdir(file_path) 
            paths = parse_directory(file_path)
        else 
            paths = [file_path]
        end 
        

        for path in paths 
            ##Open in append mode so output variables can be written 
            input_cfrad = NCDataset(path, "a")
            cfrad_dims = (input_cfrad.dim["range"], input_cfrad.dim["time"])

            ###Will generally NOT return Y, but only (X, indexer)
            ###Todo: What do I need to do for parsed args here 
            println("\r\nPROCESSING: $(path)")
            starttime=time()
            X, Y, indexer = process_single_file(input_cfrad, config_file_path; REMOVE_HIGH_PGG = true, REMOVE_LOW_NCP = true, remove_variable=indexer_var)
            println("\r\nCompleted in $(time()-starttime ) seconds")
            ##Load saved RF model 
            ##assume that default SYMBOL for saved model is savedmodel
            new_model = joblib.load(model_path)
            predictions = pyconvert(Vector{Float64}, new_model.predict(X))

            ##QC each variable in VARIALBES_TO_QC
            for var in VARIABLES_TO_QC

                ##Create new field to reshape QCed field to 
                NEW_FIELD = missings(Float64, cfrad_dims) 

                ##Only modify relevant data based on indexer, everything else should be fill value 
                QCED_FIELDS = input_cfrad[var][:][indexer]

                NEW_FIELD_ATTRS = Dict(
                    "units" => input_cfrad[var].attrib["units"],
                    "long_name" => "Random Forest Model QC'ed $(var) field"
                )

                ##Set MISSINGS to fill value in current field
                QCED_FIELDS[[ismissing(x) for x in QCED_FIELDS]] .= FILL_VAL  
                initial_count = count(QCED_FIELDS .!== FILL_VAL )
                ##Apply predictions from model 
                ##If model predicts 1, this indicates a prediction of meteorological data 
                QCED_FIELDS = map(x -> Bool(predictions[x[1]]) ? x[2] : FILL_VAL, enumerate(QCED_FIELDS))
                final_count = count(QCED_FIELDS .!== FILL_VAL)
                
                ###Need to reconstruct original 
                NEW_FIELD = NEW_FIELD[:]
                NEW_FIELD[indexer] = QCED_FIELDS
                NEW_FIELD = reshape(NEW_FIELD, cfrad_dims)
                

                try 
                    defVar(input_cfrad, var * QC_suffix, NEW_FIELD, ("range", "time"), fillvalue = FILL_VAL; attrib=NEW_FIELD_ATTRS)
                catch e
                    ###Simply overwrite the variable 
                    if e.msg == "NetCDF: String match to name in use"
                        println("Already exists... overwriting") 
                        input_cfrad[var*QC_suffix][:,:] = NEW_FIELD 
                    else 
                        throw(e)
                    end 
                end 

                
                println()
                printstyled("REMOVED $(initial_count - final_count) PRESUMED NON-METEORLOGICAL DATAPOINTS\n", color=:green)
                println("FINAL COUNT OF DATAPOINTS IN $(var): $(final_count)")

            end 
        end 
    end 



        """
    Function to split a given directory or set of directories into training and testing files using the configuration
    described in DesRosiers and Bell 2023 

    Arguments: 
        DIR_PATHS: Vector{String} consisting of directory or list of directories containing input CFRadial files 
        TRAINING_PATH: String - path to directory where training files will be linked 
        TESTING_PATH: String - path to directory where testing files will be linked

    Configuration info: 80/20 Training/Testing split, with testing cases taken from the beginning, middle, and end of 
        EACH case in DIR_PATHS. The function calculates a number of testing scans and divides it up amongst the cases, 
        so each case will be equally represented in the testing dataset. 

        In the example given in the paper, there are 4 cases, and an 80/20 split gives a total number of testing scans 
        of 356. Divided by 4, this results in 89 scans from each case. 

        Details about rounding are contained within the function

    IMPORTANT: ASSUMES SCANS FOLLOW CFRAD NAMING CONVENTIONS AND ARE THEREFORE LISTED CHRONOLOGICALLY
    """
    function split_training_testing!(DIR_PATHS::Vector{String}, TRAINING_PATH::String, TESTING_PATH::String)

        ###TODO  - make sure to ignore .tmp_hawkedit files OTHERWISE WON'T WORK AS EXPECTED 
        TRAINING_FRAC::Float32 = .72
        VALIDATION_FRAC::Float32 = .08
        TESTING_FRAC:: Float32 = 1 - TRAINING_FRAC - VALIDATION_FRAC

        ###Assume that each directory represents a different case 
        NUM_CASES::Int64 = length(DIR_PATHS)

        ###Do a little input sanitaiton
        if TRAINING_PATH[end] != '/'
            TRAINING_PATH = TRAINING_PATH * '/'
        end

        if TESTING_PATH[end] != '/'
            TESTING_PATH = TESTING_PATH * '/'
        end 

        for (i, path) in enumerate(DIR_PATHS)
            if path[end] != '/'
                DIR_PATHS[i] = path * '/'
            end
        end 

        ###Clean directories and remake them 
        rm(TESTING_PATH, force = true, recursive = true)
        rm(TRAINING_PATH, force = true, recursive = true) 

        mkdir(TESTING_PATH)
        mkdir(TRAINING_PATH) 


        TOTAL_SCANS::Int64 = 0 

        ###Calculate total number of TDR scans 
        for path in DIR_PATHS
            TOTAL_SCANS += length(readdir(path))
        end 

        ###By convention, we will round the number of training scans down 
        ###and the number of testing scans up 
        TRAINING_SCANS::Int64 = Int(floor(TOTAL_SCANS * (TRAINING_FRAC + VALIDATION_FRAC)))
        TESTING_SCANS::Int64  = Int(ceil(TOTAL_SCANS * (TESTING_FRAC)))

        ###Further by convention, will add the remainder on to the last case 
        ###A couple of notes here: Each case must have a minimum of NUM_TESTING_SCANS_PER_CASE
        ###in order to ensure each case is represented preportionally 
        ###This will be the number of scans removed, and the rest from the case will be placed into training
        NUM_TRAINING_SCANS_PER_CASE::Int64 = TRAINING_SCANS ÷ NUM_CASES
        TRAINING_REMAINDER::Int64          = TRAINING_SCANS % NUM_CASES 

        NUM_TESTING_SCANS_PER_CASE::Int64 = TESTING_SCANS ÷ NUM_CASES
        TESTING_REMAINDER::Int64          = TESTING_SCANS % NUM_CASES 


        printstyled("\nTOTAL NUMBER OF TDR SCANS ACROSS ALL CASES: $TOTAL_SCANS\n", color=:green)
        printstyled("TESTING SCANS PER CASE $(NUM_TESTING_SCANS_PER_CASE)\n", color=:orange)

        ###Each sequence of chronological TDR scans will be split as follows
        ###[[T E S T][T   R   A   I   N][T E S T][T   R   A   I   N][T E S T]]
        for path in DIR_PATHS

            contents = readdir(path)
            num_cfrads = length(contents) 

            printstyled("NUMBER OF SCANS IN CASE: $(num_cfrads)\n", color=:red)
            ###Take 1/3rd of NUM_TESTING_SCANS_PER_CASE from beginning, 1/3rd from middle, and 1/3rd from end 
            ###Need to assume files are ordered chronologically in contents here
            num_scans_for_training = num_cfrads - NUM_TESTING_SCANS_PER_CASE

            ###Need to handle a training group size that is odd 
            training_group_size = num_scans_for_training ÷ 2 
            training_group_remainder = num_scans_for_training % 2
            printstyled("TRAINING GROUP SIZE: $(training_group_size) + REMAINDER: $(training_group_remainder)\n", color=:red)

            ###If the testing_group_size is not divisible by 3, simply take the remainder from the front end (again, by definiton)
            testing_group_size = NUM_TESTING_SCANS_PER_CASE ÷ 3 
            testing_remainder = NUM_TESTING_SCANS_PER_CASE % 3
            printstyled("TESTING GROUP SIZE: $(testing_group_size) + REMAINDER $(testing_remainder)\n", color=:red)

            ###We will construct an indexer to determine which files are testing files and which 
            ###files are training files 
            testing_indexer = fill(false, num_cfrads)
            
            ###curr_idx holds the index of the LAST assignment made 
            curr_idx = 0

            ###handle first group of testing cases
            testing_indexer[1:testing_group_size + testing_remainder] .= true 
            curr_idx = testing_group_size + testing_remainder
            printstyled("\n INDEXES 1 TO $(curr_idx) ASSIGNED TESTING", color=:green)

            ###Add one group of training files
            ###Handle possible remainder here too 
            printstyled("\n INDEXES $(curr_idx) ", color=:green) 
            curr_idx = curr_idx + training_group_size + training_group_remainder 
            printstyled(" TO $(curr_idx) ASSIGNED TRAINING", color=:green) 

            ###Next group of testing files 
            printstyled("\n INDEXES $(curr_idx + 1)", color=:green)
            testing_indexer[curr_idx + 1: curr_idx + testing_group_size] .= true 
            curr_idx = curr_idx + testing_group_size 
            printstyled(" TO $(curr_idx) ASSIGNED TESTING", color=:green)

            ###Final group of training files 
            printstyled("\n INDEXES $(curr_idx + 1)", color=:green)
            curr_idx = curr_idx + training_group_size 
            printstyled(" TO $(curr_idx) ASSIGNED TRAINING", color=:green)   

            ###Final group of testing files 
            printstyled("\n INDEXES $(curr_idx + 1)", color=:green)
            testing_indexer[curr_idx + 1: curr_idx + testing_group_size] .= true 
            curr_idx = curr_idx + testing_group_size
            printstyled(" TO $(curr_idx) ASSIGNED TESTING", color=:green)

            ###Everyting not in testing will be in training 
            testing_files = contents[testing_indexer]
            training_files = contents[.!testing_indexer] 

            printstyled("\nTotal length of case files: $(num_cfrads)\n", color=:red)
            printstyled("Length of testing files: $(length(testing_files)) - $( (length(testing_files) / (num_cfrads)) ) percent\n" , color=:blue)
            printstyled("Length of testing_files: $(length(training_files)) - $( (length(training_files) / (num_cfrads)) ) percent\n", color=:blue)

            @assert (length(testing_files) + length(training_files) == num_cfrads)
            
            #printstyled("\n SßUM OF TESTING AND TRAINING = $(length(testing_files) + length(training_files))\n",color=:green)
            for file in training_files
                symlink((path * file), TRAINING_PATH * file)
            end 

            for file in testing_files
                symlink((path * file), TESTING_PATH * file)
            end
        end 

    end 

end
