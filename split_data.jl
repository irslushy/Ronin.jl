TRAINING_FRAC::Float32 = .72
VALIDATION_FRAC::Float32 = .08
TESTING_FRAC:: Float32 = 1 - TRAINING_FRAC - VALIDATION_FRAC

DIR_PATHS::Vector{String} = ["./CFRADIALS/CASES/VORTEX/", "./CFRADIALS/CASES/BAMEX/", "./CFRADIALS/CASES/RITA/", "./CFRADIALS/CASES/HAGUPIT/"]

for path in DIR_PATHS
    contents = readdir(path)
    split_location = ceil(Int32, length(contents) * (TRAINING_FRAC + VALIDATION_FRAC))

    training_files = contents[1:split_location]
    testing_files = contents[split_location + 1:end]

    printstyled("\nLength of trainign_files: $(length(training_files)) out of $(length(contents))\n", color=:blue)
    printstyled("Length of testing_files: $(length(testing_files)) out of $(length(contents))\n", color=:blue)

    if isdir(path * "TESTING")
        for file_name in testing_files
            symlink(path * file_name, "$(path)TESTING/$(file_name)")
        end
    else 
        mkdir(path * "TESTING")
        for file_name in testing_files
            symlink(path * file_name, "$(path)TESTING/$(file_name)")
        end
    end 

    if isdir(path * "TRAINING")
        for file_name in training_files
            symlink(path * file_name, "$(path)TRAINING/$(file_name)")
        end
    else 
        mkdir(path * "TRAINING")
        for file_name in training_files
            symlink(path * file_name, "$(path)TRAINING/$(file_name)")
        end
    end 
end 