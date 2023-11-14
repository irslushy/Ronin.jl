using ArgParse 
using NCDatasets
###Define queues for function queues
AVG_QUEUE = String[]
ISO_QUEUE = String[] 
STD_QUEUE = String[] 

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
            help = ("FILE or DIRECTORY\nDescribes whether the CFRAD_path describes a file or a director\n")
            default = "FILE"
        "--QC", "-Q"
            help = ("TRUE or FALSE\nDescribes whether the given file or files contain manually QCed fields\n")
            default = "FALSE"
    end

    return parse_args(s)
end

function main()
    
    parsed_args = parse_commandline()
    println("Parsed args:")
    for (arg,val) in parsed_args
        println("  $arg  =>  $val")
    end
    print(parsed_args)
    ##Load given netCDF file 
    
    cfrad = NCDataset(parsed_args["CFRad_path"])
    valid_vars = keys(cfrad)
    
    tasks = get_task_params(parsed_args["argfile"], valid_vars)
    
    fid = h5open(parsed_args["outfile"], "w")
    
    create_dataset(fid,"./X")
    attributes(fid)["Parameters"] = tasks
    
    close(fid)
end

main()

func_regex = r"(\w{1,})\((\w{1,})\)"
###Parses given parameter file and ensures that specified variables are found within the 
###passed CFradial file
###Could potentially internally return this as queues for each function 
function get_task_params(params_file, variablelist; delimiter=",")
    
    tasks = readlines(params_file)
    task_param_list = String[]
    
    for line in tasks
        if line[1] == "#"
            continue
        else
            delimited = split(line, delimiter)
            for token in delimited
                expr_ret = match(func_regex,token)
                if (typeof(expr_ret) != Nothing)
                    print("CACLULATE $(expr_ret[1]) of $(expr_ret[2])\n")
                else
                    if token in variablelist
                        push!(task_param_list, token)
                    else
                        print("\"$token\" NOT FOUND IN CFRAD FILE.... CONTINUING...\n")
                    end
                end
            end
        end 
    end 
    
    return(task_param_list)
end 
main()