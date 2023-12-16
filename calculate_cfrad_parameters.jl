include("utils.jl")
using .JMLQC_utils
using NCDatasets
using ArgParse 
using HDF5 

###TODO: Get rid of MISSING values because otherwise the model breaks - and we don't want to train it on gates that
###are already obviously missing/nonweather data

###Prefix of functions for calculating spatially averaged variables in the utils.jl file 
func_prefix = "calc_"
func_regex = r"(\w{1,})\((\w{1,})\)"

###Define queues for function queues
AVG_QUEUE = String[]
ISO_QUEUE = String[] 
STD_QUEUE = String[] 

###List of valid options for the config file
valid_funcs = ["AVG", "ISO", "STD", "AHT", "PGG"]

FILL_VAL = -32000.

RADAR_FILE_PREFIX = "cfrad"

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

function parse_directory(dir_path::String)

    paths = Vector{String}
    paths = readdir(dir_path)

    task_paths = Vector{String}()

    for path in paths
        if path[1:length(RADAR_FILE_PREFIX)] != RADAR_FILE_PREFIX
            println("ERROR: POTENTIALLY INVALID FILE FORMAT FOR FILE: $(path)")
            continue
        end 
        push!(task_paths, path)
    end
    return task_paths 
end 


#perhaps fold some of main into a function 

###Returns X features array, Y Class array, and INDEXER
###Indexer dscribes where in the scan contains missing data and where does not 
function process_file(filepath::String) 

    println("PROCCESING: $(filepath)")
    cfrad = NCDataset(filepath)
    cfrad_dims = (cfrad.dim["range"], cfrad.dim["time"])

    println("\nDIMENSIONS: $(cfrad.dims[1]) times x $(cfrad.dims[2]) ranges\n")

    valid_vars = keys(cfrad)
    tasks = get_task_params(parsed_args["argfile"], valid_vars)

    ###Features array 
    X = Matrix{Float64}(undef,cfrad.dim["time"] * cfrad.dim["range"], length(tasks))
    
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
            filled = [ismissing(x) ? Float64(FILL_VAL) : Float64(x) for x in raw]

            #any(isnan, filled) ? throw("NAN ERROR") :

            X[:, i] = filled[:]
            calc_length = time() - startTime
            println("Completed in $calc_length s"...)
            println() 
            
        else 
            println("GETTING: $task...")

            if (task == "PGG") 
                startTime = time() 
                ##Change missing values to FILL_VAL 
                X[:, i] = [ismissing(x) ? Float64(FILL_VAL) : Float64(x) for x in JMLQC_utils.calc_pgg(cfrad)[:]]
                calc_length = time() - startTime
                println("Completed in $calc_length s"...)
            elseif (task == "NCP")
                startTime = time()
                X[:, i] = [ismissing(x) ? Float64(FILL_VAL) : Float64(x) for x in JMLQC_utils.get_NCP(cfrad)[:]]
                calc_length = time() - startTime
                println("Completed in $calc_length s"...)
            else
                startTime = time() 
                X[:,i] = [ismissing(x) ? Float64(FILL_VAL) : Float64(x) for x in cfrad[task][:]]
                calc_length = time() - startTime
                println("Completed in $calc_length s"...)
            end 
            println()
        end 
    end


    println("REMOVING MISSING DATA BASED ON VT...")
    starttime = time() 
    VT = cfrad["VT"][:]
    ##@TODO check to ensure that casting is necesssary here 
    INDEXER = map((x) -> Bool(x), [ismissing(x) ? 0 : 1 for x in VT])
    println("COMPLETED IN $(round(time()-startTime, sigdigits=4))s")
    println("") 

    println("INDEXER SHAPE: $(size(INDEXER))")
    println("X SHAPE: $(size(X))")
    #println("Y SHAPE: $(size(Y))")

    X = X[INDEXER, :]
    
   

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
        println("NOT IMPLEMENTED YET") 
        return 
    else 
        paths = parsed_args["CFRad_path"]
    end 

    println("PATHS: $(paths)")
    
    ###Open specfieid dataset and determine its dimensions, 
    ###Open first dataset in the path list to get general information about the cfrad
    ###as well as the variables contained within it 
    cfrad = NCDataset(paths[1])
    cfrad_dims = (cfrad.dim["range"], cfrad.dim["time"])

    println("\nDIMENSIONS: $(cfrad.dim["time"]) times x $(cfrad.dim["range"]) ranges\n")

    ###Setup h5 file for outputting mined parameters
    ###processing will proceed in order of the tasks, so 
    ###add these as an attribute akin to column headers in the H5 dataset
    ###Also specify the fill value used 
    ###This is specified from the config file, so will be the same even when 
    ###processing in directory mode 
    fid = h5open(parsed_args["outfile"], "w")
    attributes(fid)["Parameters"] = tasks
    attributes(fid)["MISSING_FILL_VALUE"] = FILL_VAL

    ###Large features array 

    ##Dilemma here to ask Michael about: In order to determine the exact size of these arrays, we'd have to 
    ##Loop through each file and figure out the size of the missing data - this would obviously be expensive
    ##Because IO and because we'd then have to keep a large variety of separate arrays in memory. Therefore, 
    ##I'm guessing this is teh best way to do this, but open to suggestions 
    X = Matrix{Float64}
    Y = Matrix{Float64} 
    length_scan = cfrad_dims[1] * cfrad_dims[2] 

    for (i, path) in enumerate(paths)
        (newX, newY, indexer) = process_file(path) 
        X[i * length_scan:(i+1) * length_scan,:] = newX

    end


    ###Get verification information 
    ###0 indicates NON METEOROLOGICAL data that was removed during manual QC
    ###1 indicates METEOROLOGICAL data that was retained during manual QC 
    

    ###Filter dataset to remove missing VTs 
    ###Not possible to do beforehand because spatial information 
    ###Needs to be retained for some of the parameters--

    ###Use VT for filtering 
    

    if any(isnan, X)
        throw("ERROR: NaN found in features array")
    end 
    
    ##Probably only want to write once, I/O is very slow 
    write_dataset(fid, "X", X)
    write_dataset(fid, "Y", Y)
    close(fid)

end

main()