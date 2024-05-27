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
    using MLJ, MLJLinearModels, CategoricalArrays
    using PythonCall
    using DataFrames

    export get_NCP, airborne_ht, prob_groundgate
    export calc_avg, calc_std, calc_iso, process_single_file 
    export parse_directory, get_num_tasks, get_task_params, remove_validation 
    export calculate_features
    export split_training_testing! 
    export train_model 
    export QC_scan 
    export predict_with_model, evaluate_model, get_feature_importance


    """

    Function to process a set of cfradial files and produce input features for training/evaluating a model 

    # Required arguments 

    ```julia
    input_loc::String
    ```
    
    Path to input cfradial or directory of input cfradials 

    ```julia
    argument_file::String
    ```

    Path to configuration file containing which features to calculate 

    ```julia
    output_file::String
    ```

    Path to output calculated features to (generally ends in .h5)

    ```julia
    HAS_MANUAL_QC::Bool
    ```
    Specifies whether or not the file(s) have already undergone a manual QC procedure. 
    If true, function will also output a `Y` array used to verify where manual QC removed gates. This array is
    formed by considering where gates with non-missing data in raw scans (specified by `remove_variable`) are
    set to missing after QC is performed. 

    # Optional keyword arguments 

    ```julia
    verbose::Bool
    ```
    If true, will print out timing information as each file is processed 

    ```julia
    REMOVE_LOW_NCP::Bool
    ```

    If true, will ignore gates with Normalized Coherent Power/Signal Quality Index below a threshold specified in RQCFeatures.jl

    ```julia
    REMOVE_HIGH_PGG::Bool
    ```

    If true, will ignore gates with Probability of Ground Gate (PGG) values at or above a threshold specified in RQCFeatures.jl 

    ```julia
    QC_variable::String
    ```
    Name of variable in input NetCDF files that has been quality-controlled. 

    ```julia
    remove_variable::String
    ```

    Name of a raw variable in input NetCDF files. Used to determine where missing data exists in the input sweeps. 
    Data at these locations will be removed from the outputted features. 

    ```
    replace_missing 
    ```
    Whether or not to replace MISSING values with FILL_VAL in spatial parameter calculations
    Default value: False 
    """
    function calculate_features(input_loc::String, argument_file::String, output_file::String, HAS_MANUAL_QC::Bool; 
        verbose::Bool=false, REMOVE_LOW_NCP::Bool = false, REMOVE_HIGH_PGG::Bool = false, QC_variable::String = "VG", remove_variable::String = "VV", 
        replace_missing = false, write_out=true)

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
                                            REMOVE_HIGH_PGG = REMOVE_HIGH_PGG, QC_variable = QC_variable, remove_variable = remove_variable, 
                                            replace_missing=replace_missing)
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
            end

            X = vcat(X, newX)::Matrix{Float64}
            Y = vcat(Y, newY)::Matrix{Int64}

        end 
    
        println("COMPLETED PROCESSING $(length(paths)) FILES IN $(round((time() - starttime), digits = 2)) SECONDS")
    
        ###Get verification information 
        ###0 indicates NON METEOROLOGICAL data that was removed during manual QC
        ###1 indicates METEOROLOGICAL data that was retained during manual QC 
        
        ##Probably only want to write once, I/O is very slow 
        if write_out
            println()
            println("WRITING DATA TO FILE OF SHAPE $(size(X))")
            X
            println("X TYPE: $(typeof(X))")
            write_dataset(fid, "X", X)
            write_dataset(fid, "Y", Y)
            close(fid)
        else
            close(fid)
            return X, Y
        end 

    end 


    """

    Function to process a set of cfradial files and produce input features for training/evaluating a model. 
        Allows for user-specified tasks and weight matrices, otherwise the same as above.  

    # Required arguments 

    ```julia
    input_loc::String
    ```
    
    Path to input cfradial or directory of input cfradials 

    ```julia
    tasks::Vector{String}
    ```

    Vector containing the features to be calculated for each cfradial. Example `[DBZ, ISO(DBZ)]`

    ```julia
    weight_matrixes::Vector{Matrix{Union{Missing, Float64}}}
    ```

    For each task, a weight matrix specifying how much each gate in a spatial calculation will be given. 
    Required to be the same size as `tasks`

    ```julia
    output_file::String 
    ```
    
    Location to output the calculated feature data to. 

    ```julia
    HAS_MANUAL_QC::Bool
    ```
    Specifies whether or not the file(s) have already undergone a manual QC procedure. 
    If true, function will also output a `Y` array used to verify where manual QC removed gates. This array is
    formed by considering where gates with non-missing data in raw scans (specified by `remove_variable`) are
    set to missing after QC is performed. 

    # Optional keyword arguments 

    ```julia
    verbose::Bool
    ```
    If true, will print out timing information as each file is processed 

    ```julia
    REMOVE_LOW_NCP::Bool
    ```

    If true, will ignore gates with Normalized Coherent Power/Signal Quality Index below a threshold specified in RQCFeatures.jl

    ```julia
    REMOVE_HIGH_PGG::Bool
    ```

    If true, will ignore gates with Probability of Ground Gate (PGG) values at or above a threshold specified in RQCFeatures.jl 

    ```julia
    QC_variable::String
    ```
    Name of variable in input NetCDF files that has been quality-controlled. 

    ```julia
    remove_variable::String
    ```

    Name of a raw variable in input NetCDF files. Used to determine where missing data exists in the input sweeps. 
    Data at these locations will be removed from the outputted features. 

    ```
    replace_missing 
    ```
    Whether or not to replace MISSING values with FILL_VAL in spatial parameter calculations
    Default value: False 
    """
    function calculate_features(input_loc::String, tasks::Vector{String}, weight_matrixes::Vector{Matrix{Union{Missing, Float64}}}
        ,output_file::String, HAS_MANUAL_QC::Bool; verbose=false,
         REMOVE_LOW_NCP = false, REMOVE_HIGH_PGG = false, QC_variable = "VG", remove_variable = "VV", 
         replace_missing=false, write_out=true)

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
        attributes(fid)["Parameters"] = tasks
        attributes(fid)["MISSING_FILL_VALUE"] = FILL_VAL
    
        ##Instantiate Matrixes to hold calculated features and verification data 
        output_cols = length(tasks)
    
        newX = X = Matrix{Float64}(undef,0,output_cols)
        newY = Y = Matrix{Int64}(undef, 0,1) 
    
        
        starttime = time() 
    
        for (i, path) in enumerate(paths) 
            try 
                cfrad = Dataset(path) 
                pathstarttime=time() 
                (newX, newY, indexer) = process_single_file(cfrad, tasks, weight_matrixes; 
                                            HAS_MANUAL_QC = HAS_MANUAL_QC, REMOVE_LOW_NCP = REMOVE_LOW_NCP, 
                                            REMOVE_HIGH_PGG = REMOVE_HIGH_PGG, QC_variable = QC_variable, remove_variable = remove_variable, 
                                            replace_missing=replace_missing)
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
            end 

            X = vcat(X, newX)::Matrix{Float64}
            Y = vcat(Y, newY)::Matrix{Int64}

        end 
    
        println("COMPLETED PROCESSING $(length(paths)) FILES IN $(round((time() - starttime), digits = 2)) SECONDS")
    
        ###Get verification information 
        ###0 indicates NON METEOROLOGICAL data that was removed during manual QC
        ###1 indicates METEOROLOGICAL data that was retained during manual QC 
        
        ##Probably only want to write once, I/O is very slow 
        if write_out
            println()
            println("WRITING DATA TO FILE OF SHAPE $(size(X))")
            X
            println("X TYPE: $(typeof(X))")
            write_dataset(fid, "X", X)
            write_dataset(fid, "Y", Y)
            close(fid)
        else
            close(fid)
            return X, Y
        end 

    end 


    """

    Function to train a random forest model using a precalculated set of input and output features (usually output from 
    `calculate_features`). Returns nothing. 

    # Required arguments 
    ```julia
    input_h5::String
    ```
    Location of input features/targets. Input features are expected to have the name "X", and targets the name "Y". This should be 
    taken care of automatically if they are outputs from `calculate_features`

    ```julia
    model_location::String 
    ```
    Path to save the trained model out to. Typically should end in `.joblib`
    
    # Optional keyword arguments 
    ```julia
    verify::Bool = false 
    ```
    Whether or not to output a separate .h5 file containing the trained models predictions on the training set 
    (`Y_PREDICTED`) as well as the targets for the training set (`Y_ACTUAL`) 

    ```julia
    verify_out::String="model_verification.h5"
    ```
    If `verify`, the location to output this verification to. 
    """
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


    """
    Simple function that opens a given h5 file with feature data and applies a specific model to it. 
    Returns a tuple of `predictions, targets`. 

    # Required arguments 
    ```julia
    model_path::String 
    ```
    Location of trained RF model (saved in joblib file format) 

    ```julia
    input_h5::String 
    ```
    Location of h5 file containing input features. 

    """
    function predict_with_model(model_path::String, input_h5::String)
        joblib = pyimport("joblib")
        input_h5 = h5open(input_h5)
        X = input_h5["X"][:,:]
        Y = input_h5["Y"][:]
        
        new_model = joblib.load(model_path)
        predictions = pyconvert(Vector{Float64}, new_model.predict(X))
        return((predictions, Y))
    end 

    ###TODO: Fix arguments etc 
    ###Can have one for a single file and one for a directory 
    """
    Primary function to apply a trained RF model to certain raw fields of a cfradial scan. Values determined to be 
    non-meteorological by the RF model will be replaced with `Missing`

    # Required Arguments 
    ```julia
    file_path::String 
    ```
    Location of input cfradial or directory of cfradials one wishes to apply QC to 

    ```julia 
    config_file_path::String 
    ```
    Location of config file containing features to calculate as inputs to RF model 

    ```julia
    model_path::String 
    ```
    Location of trained RF model (in joblib file format) 

    # Optional Arguments 
    ```julia
    VARIABLES_TO_QC::Vector{String} = ["ZZ", "VV"]
    ```
    List containing names of raw variables in the CFRadial to apply QC algorithm to. 

    ```julia
    QC_suffix::String = "_QC"
    ```
    Used for naming the QC-ed variables in the modified CFRadial file. Field name will be QC_suffix appended to the raw field. 
    Example: `DBZ_QC`

    ```julia
    indexer_var::String = "VV"
    ```
    Variable used to determine what gates are considered "missing" in the raw moments. QC will not 
    be applied to these gates, they will simply remain missing. 

    ```julia
    decision_threshold::Float64 = .5
    ```
    Used to leverage probablistic nature of random forest methodology. When the model has a greater than `decision_threshold`
    level confidence that a gate is meteorological data, it will be assigned as such. Anything at or below this confidence threshold
    will be assigned non-meteorological. At least in the ELDORA case, aggressive thresholds (.8 and above) have been found to maintain 
    >92% of the meteorological data while removing >99% of non-meteorological gates. 

    """
    function QC_scan(file_path::String, config_file_path::String, model_path::String; VARIABLES_TO_QC::Vector{String}= ["ZZ", "VV"],
                     QC_suffix::String = "_QC", indexer_var::String="VV", decision_threshold::Float64 = .5)
        
        joblib = pyimport("joblib") 
        new_model = joblib.load(model_path)

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
            ##For binary classifications, 1 will be at index 2 in the predictions matrix 
            met_predictions = pyconvert(Matrix{Float64}, new_model.predict_proba(X))[:, 2]
            predictions = met_predictions .> decision_threshold

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
                
                initial_count = count(.!map(ismissing, QCED_FIELDS))
                ##Apply predictions from model 
                ##If model predicts 1, this indicates a prediction of meteorological data 
                QCED_FIELDS = map(x -> Bool(predictions[x[1]]) ? x[2] : missing, enumerate(QCED_FIELDS))
                final_count = count(.!map(ismissing, QCED_FIELDS))
                
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
            
            close(input_cfrad)

        end 
    end 



    """
    Function to split a given directory or set of directories into training and testing files using the configuration
    described in DesRosiers and Bell 2023. **This function assumes that input directories only contain cfradial files 
    that follow standard naming conventions, and are thus implicitly chronologically ordered.** The function operates 
    by first dividing file names into training and testing sets following an 80/20 training/testing split, and subsequently
    softlinking each file to the training and testing directories. Attempts to avoid temporal autocorrelation while maximizing 
    variance by dividing each case into several different training/testing sections. 

    # Required Arguments: 

    ```julia
    DIR_PATHS::Vector{String}
    ```
    List of directories containing cfradials to be used for model training/testing. Useful if input data is split 
    into several different cases. 

    ```julia
    TRAINING_PATH::String 
    ```
    Directory to softlink files designated for training into. 

    ```julia
    TESTING_PATH::String 
    ```
    Directory to softlink files designated for testing into. 
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





    """

    Function that takes in a given model, directory containing either cfradials or an already processed h5 file, 
    a path to the configuration file, and a mode type ("C" for cfradials "H" for h5) and returns a Julia DataFrame 
    containing a variety of metrics about the model's performance on the specified set, including precision and recall scores. 

    # Required arguments 
    ```julia
    model_path::String
    ```

    Path to input trained random forest model

    ```julia
    input_file_dir::String
    ```

    Path to input h5 file or directory of cfradial files to be processed

    ```julia
    config_file_path 
    ```
    
    Path to configuration file containing information about what features to calculate 

    # Optional Arguments 
    
    ```julia
    mode::String = "C"
    ```
    Whether to process a directory of cfradial files ("C" mode) or simply utilize an already-processed h5 file ("H" mode) 

    ```julia
    write_out::Bool = false
    ```
    If in "C" mode, whether or not to write the resulting calculated features out to a file 

    ```julia
    output_file::String = "_.h5" 
    ```

    Location to write calculated output features to. 

    Also contains all keyword arguments for calculate_features 
    
    """
    function evaluate_model(model_path::String, input_file_dir::String, config_file_path::String; mode="C",
        HAS_MANUAL_QC=false, verbose=false, REMOVE_LOW_NCP=false, REMOVE_HIGH_PGG=false, 
        QC_variable="VG", remove_variable = "VV", replace_missing = false, output_file = "_.h5", write_out=false)

        joblib = pyimport("joblib") 
        curr_model = joblib.load(model_path)

    if mode == "C"

        X, Y = calculate_features(input_file_dir, config_file_path, output_file=output_file, HAS_MANUAL_QC = HAS_MANUAL_QC 
                        ; verbose=verbose, REMOVE_LOW_NCP=REMOVE_LOW_NCP, REMOVE_HIGH_PGG=REMOVE_HIGH_PGG, 
                        QC_variable=QC_variable, remove_variable=remove_variable, replace_missing=replace_missing,
                        write_out=write_out )


        probs = pyconvert(Matrix{Float64}, curr_model.predict_proba(X))

    elseif mode == "H"

        input_h5 = h5open(input_file_dir)

        X = input_h5["X"][:,:]
        Y = input_h5["Y"][:,:]

        close(input_h5) 

        probs = pyconvert(Matrix{Float64}, curr_model.predict_proba(X))

    else 
        print("ERROR: UNKNOWN MODE")
    end 

    ###Now, iterate through probabilities and calculate predictions for each one. 
    proba_seq = .1:.1:.9
    met_probs = probs[:, 2]


    list_names = ["prob_level", "true_pos", "false_pos", "true_neg", "false_neg", "prec", "rec", "removal"]
    for name in (Symbol(_) for _ in list_names)
        @eval $name = []
    end 

    for prob in proba_seq

        met_predictions = met_probs .>= prob 
        print(length(met_predictions))
        print(length(Y))
        tpc = count(Y[met_predictions .== 1] .== met_predictions[met_predictions .== 1])
        fpc = count(Y[met_predictions .== 1] .!= met_predictions[met_predictions .== 1])

        push!(true_pos, tpc)
        push!(false_pos, fpc)

        tnc =  count(Y[met_predictions .== 0] .== met_predictions[met_predictions .== 0])
        fnc =  count(Y[met_predictions .== 0] .!= met_predictions[met_predictions .== 0])

        push!(true_neg, tnc)
        push!(false_neg, fnc) 

        push!(prob_level, prob) 

        push!(prec, tpc / (tpc + fpc) )
        push!(rec, tpc / (tpc + fnc))

        ###Removal is the fraction of true negatives of total negatives 
        push!(removal, tnc / (tnc + fpc))

        ###I can likely make this better using some metaprogramming but it's fine for now 

        end 
        return(DataFrame(prob_level=prob_level, true_pos=true_pos, false_pos=false_pos, true_neg=true_neg, false_neg=false_neg, precision=prec,
            recall=rec, removal=removal ))
     end 



    function standardize(column)
        col_max = maximum(column)
        col_min = minimum(column)
        return (map(x-> (x - col_min) / (col_max - col_min), column))
    end 


     """
     # Uses L1 regression with a variety of λ penalty values to determine the most useful features for
     input to the random forest model. 

     ---

     # Required Input

     ---

     ```julia
     input_file_path::String 
     ```

     Path to .h5 file containing model training features under `["X"]` parameter, and model targets under `["Y"]` parameter. 
     Also expects the h5 file to contain an attribute known as `Parameters` containing abbreviations for the feature types 

    ```julia 
    λs::Vector{Float64}
    ```

    Vector of values used to vary the strength of the penalty term in the regularization. 
    ---

    # Optional Keyword Arguments 

    ---

    ```julia
    pred_threshold::Float64
    ```

    Minimum cofidence level for binary classifier when predicting 
    ---
    #Returns
    ---
    Returns a DataFrame with each row containing info about a regression for a specific λ, the values of the regression coefficients 
        for each input feature, and the Root Mean Square Error of the resultant regression. 
    """
     function get_feature_importance(input_file_path::String, λs::Vector{Float64}; pred_threshold::Float64 = .5)


        @load LogisticClassifier pkg=MLJLinearModels

        training_data = h5open(input_file_path)
        
        ###Standardize features to expedite regression convergence 
        features = mapslices(standardize, training_data["X"][:,:], dims=1)
        ###Flatten targets and convert to categorical datatime 
        targets = categorical(training_data["Y"][:,:][:])
        targets_raw = training_data["Y"][:, :][:]
        params = attrs(training_data)["Parameters"]

        close(training_data)

        coef_values = Dict(param => [] for param in params)
        coef_values["λ"] = λs 
        rmses = [] 

        for λ in λs

            mdl = LogisticClassifier(;lambda=λ, penalty=:l1)
            mach = machine(mdl, MLJ.table(features), targets[:])
            fit!(mach)
            coefs = fitted_params(mach).coefs 

            y_pred = predict(mach, features)
            results = pdf(y_pred, [0, 1])
            met_predictions = map(x -> x > pred_threshold ? 0 : 1, results[:, 1])

            push!(rmses, MLJ.rmse(met_predictions, targets_raw))

            for (i, param) in enumerate(params)
                push!(coef_values[param], coefs[i][2])
            end 
        
            
        end 

        coef_values["rmse"] = rmses
        return(DataFrame(coef_values))

    end 


end
