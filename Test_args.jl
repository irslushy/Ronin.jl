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

valid_funcs = ["AVG", "ISO", "STD", "AHT", "PGG"]

FILL_VAL = -32000.

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
            help = ("FILE or DIRECTORY\nDescribes whether the CFRAD_path describes a file or a director\n
                    DEFAULT: FILE")
            default = "FILE"
        "--QC", "-Q"
            help = ("TRUE or FALSE\nDescribes whether the given file or files contain manually QCed fields\n
                    DEFAULT: FALSE")
    end

    return parse_args(s)
end

function get_task_params(params_file, variablelist; delimiter=",")
    
    tasks = readlines(params_file)
    task_param_list = String[]
    
    for line in tasks
        if line[1] == '#'
            continue
        else
            delimited = split(line, delimiter)
            for token in delimited
                token = strip(token, ' ')
                expr_ret = match(func_regex,token)
                if (typeof(expr_ret) != Nothing)
                    if (expr_ret[1] ∉ valid_funcs || expr_ret[2] ∉ variablelist)
                        println("ERROR: CANNOT CALCULATE $(expr_ret[1]) of $(expr_ret[2])\n", 
                        "Potentially invalid function or missing variable\n")
                    else
                        ###Add λ to the front of the token to indicate it is a function call
                        push!(task_param_list, "λ" * token)
                    end 
                else
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

###Define queues for function queues

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
            help = ("FILE or DIRECTORY\nDescribes whether the CFRAD_path describes a file or a director\n
                    DEFAULT: FILE")
            default = "FILE"
        "--QC", "-Q"
            help = ("TRUE or FALSE\nDescribes whether the given file or files contain manually QCed fields\n
                    DEFAULT: FALSE")
    end

    return parse_args(s)
end




function main()
    
    parsed_args = parse_commandline()
#     println("Parsed args:")
#     for (arg,val) in parsed_args
#         println("  $arg  =>  $val")
#     end
    ##Load given netCDF file 
    
    cfrad = NCDataset(parsed_args["CFRad_path"])
    cfrad_dims = (cfrad.dim["range"], cfrad.dim["time"])
    println("\nDIMENSIONS: $cfrad_dims\n")

    valid_vars = keys(cfrad)
    
    tasks = get_task_params(parsed_args["argfile"], valid_vars)
    
    fid = h5open(parsed_args["outfile"], "w")
    attributes(fid)["Parameters"] = tasks
    attributes(fid)["MISSING_FILL_VALUE"] = FILL_VAL
    ###Features array 
    X = Matrix{Float64}(undef,cfrad.dim["time"] * cfrad.dim["range"], length(tasks))
    
    for (i, task) in enumerate(tasks)
  
        ###λ identifier indicates that the requested task is a function 
        if (task[1] == 'λ')
            
            curr_func = lowercase(task[3:5])
            var = task[7:9]
            
            println("CALCULATING $curr_func OF $var... ")
            
            curr_func = Symbol(func_prefix * curr_func)
            startTime = time() 

            raw = @eval $curr_func($cfrad[$var][:,:])[:]
            filled = Vector{Float64}
            filled = [ismissing(x) ? Float64(FILL_VAL) : Float64(x) for x in raw]
            X[:, i] = filled[:]
            calc_length = time() - startTime
            println("Completed in $calc_length s"...)
            println()
            
        else 
            println("GETTING: $task...")

            if (task == "PGG") 
                startTime = time() 
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

    ###Filter dataset to remove missing VTs 
    ###Not possible to do beforehand because spatial information 
    ###Needs to be retained for some of the parameters--
    X = X[X[:,1].!= FILL_VAL, :]
    final_shape = size(X)
    println("FINAL ARRAY SHAPE: $final_shape")
    write_dataset(fid, "X", X)
    close(fid)
end

main()


###Parses given parameter file and ensures that specified variables are found within the 
###passed CFradial file
###Could potentially internally return this as queues for each function 
