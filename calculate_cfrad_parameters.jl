include("utils.jl")
using .JMLQC_utils
using NCDatasets
using ArgParse 
using HDF5 

###TODO: Get rid of MISSING values because otherwise the model breaks - and we don't want to train it on gates that
###are already obviously missing/nonweather data

###Prefix of functions for calculating spatially averaged variables in the utils.jl file 
func_prefix::String= "calc_"
func_regex::Regex = r"(\w{1,})\((\w{1,})\)"

###List of valid options for the config file
valid_funcs::Array{String} = ["AVG", "ISO", "STD", "AHT", "PGG"]

###A note on FILL_VAL: 
###This will replace NaNs for isolated gates in the spatial parameters 

FILL_VAL::Float64 = -32000.
RADAR_FILE_PREFIX::String = "cfrad"

##Threshold to exclude gates from when <= for model training set 
NCP_THRESHOLD::Float64 = .2 
PGG_THRESHOLD::Float64 = 1. 

REMOVE_LOW_NCP::Bool = true 
REMOVE_HIGH_PGG::Bool = true 

###Set up argument table for CLI 
function parse_commandline()
    
    s = ArgParseSettings()

    @add_arg_table s begin
        
        "CFRad_path"
            help = "Path to input CFRadial File"
            required = true
        "--argfile","-f"
            help = ("File containing comma-delimited list of variables you wish the script to calculate and output\n
                    Currently supported funcs include AVG(var_name), STD(var_name), and ISO(var_name)\n
                    Example file content: DBZ, VEL, AVG(DBZ), AVG(VEL), ISO(DBZ), STD(DBZ)\n")
        "--outfile", "-o"
            help = "Location to output mined data to"
            default = "./mined_data.h5"
        "--mode", "-m" 
            help = ("F or D \nDescribes whether the CFRAD_path describes a file (F) or a directory (D)\n
                    DEFAULT: F")
            default = "F"
        "--QC", "-Q"
            help = ("TRUE or FALSE\nDescribes whether the given file or files contain manually QCed fields\n
                    DEFAULT: FALSE")
    end

    return parse_args(s)
end

###Parses the specified parameter file 
###Thinks of the parameter file as a list of tasks to perform on variables in the CFRAD
function get_task_params(params_file, variablelist; delimiter=",")
    
    tasks = readlines(params_file)
    task_param_list = String[]
    
    for line in tasks
        ###Ignore comments in the parameter file 
        if line[1] == '#'
            continue
        else
            delimited = split(line, delimiter)
            for token in delimited
                ###Remove whitespace and see if the token is of the form ABC(DEF)
                token = strip(token, ' ')
                expr_ret = match(func_regex,token)
                ###If it is, make sure that it is both a valid function and a valid variable 
                if (typeof(expr_ret) != Nothing)
                    if (expr_ret[1] ∉ valid_funcs || expr_ret[2] ∉ variablelist)
                        println("ERROR: CANNOT CALCULATE $(expr_ret[1]) of $(expr_ret[2])\n", 
                        "Potentially invalid function or missing variable\n")
                    else
                        ###Add λ to the front of the token to indicate it is a function call
                        ###This helps later when trying to determine what to do with each "task" 
                        push!(task_param_list, "λ" * token)
                    end 
                else
                    ###Otherwise, check to see if this is a valid variable 
                    if token in variablelist || token ∈ valid_funcs
                        push!(task_param_list, token)
                    else
                        println("\"$token\" NOT FOUND IN CFRAD FILE.... CONTINUING...\n")
                    end
                end
            end
        end 
    end 
    
    return(task_param_list)
end 

function get_num_tasks(params_file; delimeter = ",")

    tasks = readlines(params_file)
    num_tasks = 0 

    for line in tasks
        if line[1] == '#'
            continue
        else 
            delimited = split(line, delimeter) 
            num_tasks = num_tasks + length(delimited)
        end
    end 

    return num_tasks
end

function parse_directory(dir_path::String)

    paths = Vector{String}
    paths = readdir(dir_path)

    task_paths = Vector{String}()

    for path in paths
        if path[1:length(RADAR_FILE_PREFIX)] != RADAR_FILE_PREFIX
            println("ERROR: POTENTIALLY INVALID FILE FORMAT FOR FILE: $(path)")
            continue
        end 
        push!(task_paths, dir_path * "/" * path)
    end
    return task_paths 
end 


#perhaps fold some of main into a function 

###Returns X features array, Y Class array, and INDEXER
###Indexer dscribes where in the scan contains missing data and where does not 
function process_file(filepath::String, parsed_args)

    println("PROCCESING: $(filepath)")
    cfrad = NCDataset(filepath)
    cfrad_dims = (cfrad.dim["range"], cfrad.dim["time"])

    println("\nDIMENSIONS: $(cfrad_dims[1]) times x $(cfrad_dims[2]) ranges\n")

    valid_vars = keys(cfrad)
    tasks = get_task_params(parsed_args["argfile"], valid_vars)

    ###Features array 
    X = Matrix{Float64}(undef,cfrad.dim["time"] * cfrad.dim["range"], length(tasks))

    ###Array to hold PGG for indexing  
    PGG = Matrix{Float64}(undef, cfrad.dim["time"]*cfrad.dim["range"], 1)
    PGG_Completed_Flag = false 

    for (i, task) in enumerate(tasks)
  
        ###λ identifier indicates that the requested task is a function 
        if (task[1] == 'λ')
            
            ###Need to use capturing groups again here
            curr_func = lowercase(task[3:5])
            var = match(func_regex, task)[2]
            
            println("CALCULATING $curr_func OF $var... ")
            println("TYPE OF VARIABLE: $(typeof(cfrad[var][:,:]))")
            curr_func = Symbol(func_prefix * curr_func)
            startTime = time() 

            @time raw = @eval $curr_func($cfrad[$var][:,:])[:]
            filled = Vector{Float64}
            filled = [ismissing(x) || isnan(x) ? Float64(FILL_VAL) : Float64(x) for x in raw]

            any(isnan, filled) ? throw("NAN ERROR") : 

            X[:, i] = filled[:]
            calc_length = time() - startTime
            println("Completed in $calc_length s"...)
            println() 
            
        else 
            println("GETTING: $task...")

            if (task == "PGG") 
                startTime = time() 
                ##Change missing values to FILL_VAL 
                PGG = [ismissing(x) || isnan(x) ? Float64(FILL_VAL) : Float64(x) for x in JMLQC_utils.calc_pgg(cfrad)[:]]
                X[:, i] = PGG
                PGG_Completed_Flag = true 
                calc_length = time() - startTime
                println("Completed in $calc_length s"...)
            elseif (task == "NCP")
                startTime = time()
                X[:, i] = [ismissing(x) || isnan(x) ? Float64(FILL_VAL) : Float64(x) for x in JMLQC_utils.get_NCP(cfrad)[:]]
                calc_length = time() - startTime
                println("Completed in $calc_length s"...)
            elseif (task == "AHT")
                startTime = time()
                X[:, i] = [ismissing(x) || isnan(x) ? Float64(FILL_VAL) : Float64(x) for x in JMLQC_utils.calc_aht(cfrad)[:]]
                println("Completed in $(time() - startTime) seconds")
            else
                startTime = time() 
                X[:, i] = [ismissing(x) || isnan(x) ? Float64(FILL_VAL) : Float64(x) for x in cfrad[task][:]]
                calc_length = time() - startTime
                println("Completed in $calc_length s"...)
            end 
            println()
        end 
    end


    ###Uses INDEXER to remove data not meeting basic quality thresholds
    ###A value of 0 in INDEXER will remove the data from training/evaluation 
    ###by the subsequent random forest model 
    println("REMOVING MISSING DATA BASED ON VT...")
    starttime = time() 
    VT = cfrad["VT"][:]

    INDEXER = [ismissing(x) ? false : true for x in VT]
    println("COMPLETED IN $(round(time()-starttime, sigdigits=4))s")
    println("") 

    println("FILTERING")
    starttime=time()
    ###Missings might be implicit in this already 
    if (REMOVE_LOW_NCP)
        println("REMOVING BASED ON NCP")
        println("INITIAL COUNT: $(count(INDEXER))")
        NCP = JMLQC_utils.get_NCP(cfrad)
        ###bitwise or with inital indexer for where NCP is <= NCP_THRESHOLD
        INDEXER[INDEXER] = [x <= NCP_THRESHOLD ? false : true for x in NCP[INDEXER]]
        println("FINAL COUNT: $(count(INDEXER))")
    end

    if (REMOVE_HIGH_PGG)

        println("REMOVING BASED ON PGG")
        println("INITIAL COUNT: $(count(INDEXER))")

        if (PGG_Completed_Flag)
            INDEXER[INDEXER] = [x >= PGG_THRESHOLD ? false : true for x in PGG[INDEXER]]
        else
            PGG = [ismissing(x) || isnan(x) ? Float64(FILL_VAL) : Float64(x) for x in JMLQC_utils.calc_pgg(cfrad)[:]]
            INDEXER[INDEXER] = [x >= PGG_THRESHOLD ? false : true for x in PGG[INDEXER]]
        end
        println("FINAL COUNT: $(count(INDEXER))")
        println()
    end
    println("COMPLETED IN $(round(time()-starttime, sigdigits=4))s")


    println("INDEXER SHAPE: $(size(INDEXER))")
    println("X SHAPE: $(size(X))")
    #println("Y SHAPE: $(size(Y))")

    X = X[INDEXER, :] 
    println("NEW X SHAPE: $(size(X))")
    #any(isnan, X) ? throw("NAN ERROR") : 

    println("Parsing METEOROLOGICAL/NON METEOROLOGICAL data")
    startTime = time() 

    ###Filter the input arrays first 
    VG = cfrad["VG"][:][INDEXER]
    VV = cfrad["VV"][:][INDEXER]

    Y = reshape([ismissing(x) ? 0 : 1 for x in VG .- VV][:], (:, 1))
    calc_length = time() - startTime
    println("Completed in $calc_length s"...)
    println()

    println()
    println("FINAL X SHAPE: $(size(X))")
    println("FINAL Y SHAPE: $(size(Y))")

    return(X, Y, INDEXER)
end 

function main()
    
    parsed_args = parse_commandline()
#     println("Parsed args:")
#     for (arg,val) in parsed_args
#         println("  $arg  =>  $val")
#     end
    ##Load given netCDF file 
    

    ##If this is a directory, things get a little more complicated 
    paths = Vector{String}()

    if parsed_args["mode"] == "D"
        paths = parse_directory(parsed_args["CFRad_path"])
    else 
        paths = [parsed_args["CFRad_path"]]
    end 
    
    ###Setup h5 file for outputting mined parameters
    ###processing will proceed in order of the tasks, so 
    ###add these as an attribute akin to column headers in the H5 dataset
    ###Also specify the fill value used 

    println("OUTPUT FILE $(parsed_args["outfile"])")
    fid = h5open(parsed_args["outfile"], "w")

    ##@TODO: re-add these attributes 
    ##@TODO: Figure out how to get the tasks without checking the validity of the file 
    #attributes(fid)["Parameters"] = tasks
    attributes(fid)["MISSING_FILL_VALUE"] = FILL_VAL

    ##Dilemma here to ask Michael about: In order to determine the exact size of these arrays, we'd have to 
    ##Loop through each file and figure out the size of the missing data - this would obviously be expensive
    ##Because IO and because we'd then have to keep a large variety of separate arrays in memory. Therefore, 

    ##I'm guessing this is teh best way to do this, but open to suggestions 
    output_cols = get_num_tasks(parsed_args["argfile"])
    newX = X = Matrix{Float64}(undef,0,output_cols)

    ###COULD OPTIMIZE THIS TO BITMATRIX 
    newY = Y = Matrix{Int64}(undef, 0,1) 

    ##Will always have at leaset 1 path
    #length_scan = cfrad_dims[1] * cfrad_dims[2] 
    starttime = time() 

    ###Exceptions to handle here (That I've discovered so far)
    ###Invalid task in config file 
    for (i, path) in enumerate(paths) 
        try 
            (newX, newY, indexer) = process_file(path, parsed_args)
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
    

    ###Filter dataset to remove missing VTs 
    ###Not possible to do beforehand because spatial information 
    ###Needs to be retained for some of the parameters--

    
    ##Probably only want to write once, I/O is very slow 
    println()
    println("WRITING DATA TO FILE OF SHAPE $(size(X))")
    X
    println("X TYPE: $(typeof(X))")
    write_dataset(fid, "X", X)
    write_dataset(fid, "Y", Y)
    close(fid)

end

main()