include("./src/JMLQC_utils.jl")
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
        "--output_to_directory"
            help = ("Choose whether to output h5 to a new directory of the same name as output h5 file
                     - also copies config file so one can retain information about which parameters the model is trained on")
            default = false
        "--QC", "-Q"
            help = ("TRUE or FALSE\nDescribes whether the given file or files contain manually QCed fields\n
                    DEFAULT: FALSE")
            default = false
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
    

    ##If this is a directory, things get a little more complicated 
    paths = Vector{String}()

    if uppercase(parsed_args["mode"]) == "D"
        paths = parse_directory(parsed_args["CFRad_path"])
    else 
        paths = [parsed_args["CFRad_path"]]
    end 
    
    ###Setup h5 file for outputting mined parameters
    ###processing will proceed in order of the tasks, so 
    ###add these as an attribute akin to column headers in the H5 dataset
    ###Also specify the fill value used 

    println("OUTPUTTING DATA TO FILE: $(parsed_args["outfile"])")
    fid = h5open(parsed_args["outfile"], "w")

    ###Add information to output h5 file 
    #attributes(fid)["Parameters"] = tasks
    #attributes(fid)["MISSING_FILL_VALUE"] = FILL_VAL

    ##Instantiate Matrixes to hold calculated features and verification data 
    output_cols = get_num_tasks(parsed_args["argfile"])

    newX = X = Matrix{Float64}(undef,0,output_cols)
    newY = Y = Matrix{Int64}(undef, 0,1) 

 
    starttime = time() 

    for (i, path) in enumerate(paths) 
        try 
            cfrad = Dataset(path) 
            (newX, newY, indexer) = process_single_file(cfrad, parsed_args)
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