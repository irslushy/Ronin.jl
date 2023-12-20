TRAINING_FRAC::Float32 = .72
VALIDATION_FRAC::Float32 = .08
TESTING_FRAC:: Float32 = 1 - TRAINING_FRAC - VALIDATION_FRAC

DIR_PATHS::Vector{String} = ["/Users/ischluesche/Documents/Grad_School/Research/JMLQC/CFRADIALS/CASES/VORTEX/", 
"/Users/ischluesche/Documents/Grad_School/Research/JMLQC/CFRADIALS/CASES/BAMEX/",
"/Users/ischluesche/Documents/Grad_School/Research/JMLQC/CFRADIALS/CASES/RITA/", 
"/Users/ischluesche/Documents/Grad_School/Research/JMLQC/CFRADIALS/CASES/HAGUPIT/"]

TESTING_PATH::String = "/Users/ischluesche/Documents/Grad_School/Research/JMLQC/CFRADIALS/CASES/TESTING/"
TRAINING_PATH::String = "/Users/ischluesche/Documents/Grad_School/Research/JMLQC/CFRADIALS/CASES/TRAINING/"


###Clean things
rm(TESTING_PATH, force = true, recursive = true)
rm(TRAINING_PATH, force = true, recursive = true) 

mkdir(TESTING_PATH)
mkdir(TRAINING_PATH) 

for path in DIR_PATHS

    contents = readdir(path)
    split_location = ceil(Int32, length(contents) * (TRAINING_FRAC + VALIDATION_FRAC))

    training_files = contents[1:split_location]
    testing_files = contents[split_location + 1:end]

    printstyled("\nLength of training files: $(length(training_files)) out of $(length(contents))\n", color=:blue)
    printstyled("Length of testing_files: $(length(testing_files)) out of $(length(contents))\n", color=:blue)

    for file in training_files
        symlink((path * file), TRAINING_PATH * file)
    end 

    for file in testing_files
        symlink((path * file), TESTING_PATH * file)
    end
end 