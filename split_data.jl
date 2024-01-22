
###Script designed to split directories with CFRAD files in them into the distribution descrbed in
###The DesRosiers + Bell JMLQC manuscript
###
###This splits the datapoints into a training and testing set, where the testing data 
###comprises 20% of the data and the training the other 80%. In order to increase temporal variability
###testing scans are selected from the beginning, middle, and end of their respective datasets 
###One key assumption is that the file naming is CF-compliant, and as such implicitly 
###chronologically orders the files. 

###NOTE: The script will remove the directories specified by TESTING_PATH and TRAINING_PATH
###In order to ensure that they are clean and do not contain extraneous files. 

TRAINING_FRAC::Float32 = .72
VALIDATION_FRAC::Float32 = .08
TESTING_FRAC:: Float32 = 1 - TRAINING_FRAC - VALIDATION_FRAC

###List of paths to each case directory
DIR_PATHS::Vector{String} = ["/Users/ischluesche/Documents/Grad_School/Research/JMLQC/CFRADIALS/CASES/VORTEX/", 
"/Users/ischluesche/Documents/Grad_School/Research/JMLQC/CFRADIALS/CASES/BAMEX/",
"/Users/ischluesche/Documents/Grad_School/Research/JMLQC/CFRADIALS/CASES/RITA/", 
"/Users/ischluesche/Documents/Grad_School/Research/JMLQC/CFRADIALS/CASES/HAGUPIT/"]

###Paths specifiying where testing and training files will be softlinked to 
TESTING_PATH::String = "/Users/ischluesche/Documents/Grad_School/Research/JMLQC/CFRADIALS/CASES/TESTING/"
TRAINING_PATH::String = "/Users/ischluesche/Documents/Grad_School/Research/JMLQC/CFRADIALS/CASES/TRAINING/"

###Assume that each directory represents a different case 
NUM_CASES::Int64 = length(DIR_PATHS)

###Clean things
printstyled("\nWARNING: REMOVING THE FOLLOWING DIRECTORIES:\n $(TESTING_PATH) \n $(TRAINING_PATH)\n OK? (Y/N)\n", color=:red)
resp = readline()

while (resp ∉ ["Y", "N"])
    printstyled("OK?  (Y/N)\n", color=:red)
    global resp = readline()
end 

if resp == "N"
    printstyled("\nEXITING.....\n", color=:red)
    exit() 
end 

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

###Further by convention, will add the remainder on to the last scan 
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
    num_cases = length(contents) 

    printstyled("NUMBER OF SCANS IN CASE: $(num_cases)\n", color=:red)
    ###Take 1/3rd of NUM_TESTING_SCANS_PER_CASE from beginning, 1/3rd from middle, and 1/3rd from end 
    ###Need to assume files are ordered chronologically in contents here
    num_scans_for_training = num_cases - NUM_TESTING_SCANS_PER_CASE

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
    testing_indexer = fill(false, num_cases)
    
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


    printstyled("\nTotal length of case files: $(num_cases)\n", color=:red)
    printstyled("Length of testing files: $(count(testing_indexer))\n", color=:blue)
    printstyled("Length of testing_files: $(count(.!testing_indexer))\n", color=:blue)
    
    testing_files = contents[testing_indexer]
    training_files = contents[.!testing_indexer] 

    @assert (length(testing_files) + length(training_files) == num_cases)
    #printstyled("\n SßUM OF TESTING AND TRAINING = $(length(testing_files) + length(training_files))\n",color=:green)
    for file in training_files
        symlink((path * file), TRAINING_PATH * file)
    end 

    for file in testing_files
        symlink((path * file), TESTING_PATH * file)
    end
end 