###Function containing capabilites to split input datasets into different training/testing compoments and Otherwise
###Input/output functionality 
export remove_validation 
export split_training_testing 

function remove_validation(input_dataset::String; training_output="train_no_validation_set.h5", validation_output = "validation.h5")
        
    currset = h5open(input_dataset)

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

    write_dataset(training_out, "X", X[test_indexer, :])
    write_dataset(training_out, "Y", Y[test_indexer, :])

    close(currset)
    close(validation_out)
    close(training_out)

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
function split_training_testing(DIR_PATHS::Vector{String}, TRAINING_PATH::String, TESTING_PATH::String)

    ###TODO  - make sure to ignore .tmp_hawkedit files OTHERWISE WON'T WORK AS EXPECTED 
    TRAINING_FRAC::Float32 = .72
    VALIDATION_FRAC::Float32 = .08
    TESTING_FRAC:: Float32 = 1 - TRAINING_FRAC - VALIDATION_FRAC

    ###Assume that each directory represents a different case 
    NUM_CASES::Int64 = length(DIR_PATHS)

    ###Clean directories and remake them 
    rm(TESTING_PATH, force = true, recursive = true)
    rm(TRAINING_PATH, force = true, recursive = true) 

    mkdir(TESTING_PATH)
    mkdir(TRAINING_PATH) 


    TOTAL_SCANS::Int64 = 0 

    ###Calculate total number of TDR scans 
    for path in DIR_PATHS
        global TOTAL_SCANS
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