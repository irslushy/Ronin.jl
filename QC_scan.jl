using NCDatasets
using HDF5
using MLJ
using PyCall, PyCallUtils, BSON
using ArgParse
using Missings 

include("./src/JMLQC_utils.jl")
using .JMLQC_utils 

using ScikitLearn
@sk_import ensemble: RandomForestClassifier

###Change this to apply quality control to different variables in CFRAD
VARIALBES_TO_QC::Vector{String} = ["DBZ", "VEL"]
###New variable name in netcdf file will be VARNAME * QC_SUFFIX 
QC_SUFFIX::String = "_FILTERED"

###Set up argument table for CLI 
###TODO: Add argument to decide whether to write input QC file to same CFRad or NEW CFrad 
function parse_commandline()
    
    s = ArgParseSettings()

    @add_arg_table s begin
        
        "CFRad_path"
            help = "Path to input CFRadial File"
            required = true
        "model_path"
            help = "Location of trained RF model in BSON format to load"
            required=true 
        "argfile"
            help = ("File containing comma-delimited list of variables you wish the script to calculate and output\n
                    Currently supported funcs include AVG(var_name), STD(var_name), and ISO(var_name)\n
                    Example file content: DBZ, VEL, AVG(DBZ), AVG(VEL), ISO(DBZ), STD(DBZ)\n")
            required=true 
        "-m", "--mode"
            help = "Whether to run in directory mode (D), single-scan mode(S), or 
                    Listening Mode (L). If directory mode, CFRad_path should be a directory
                    containing a variety of scans to apply QC to. If single-scan, this should
                    be a single file. If Listening, path should contain a directory. The script will
                    run until interrupted, processing new files as they are added to the directory"
            default="S"
        
        "--outfile", "-o"
            help = "Location to output mined data to"
            default = "./mined_data.h5"
        "-i"
            help = "variable to index off of"
            default = "VV"

        
    end

    return parse_args(s)
end

function main()     
    ##Could make this a try-catch block if you really wanted to be 
    ##Extra safe about it 
    parsed_args = parse_commandline() 

    ##Open in append mode so output variables can be written 
    input_cfrad = NCDataset(parsed_args["CFRad_path"], "a")
    cfrad_dims = (input_cfrad.dim["range"], input_cfrad.dim["time"])

    ###Will generally NOT return Y, but only (X, indexer)
    X, Y, indexer = JMLQC_utils.process_single_file(input_cfrad, parsed_args; REMOVE_HIGH_PGG = true, REMOVE_LOW_NCP = true, remove_variable=parsed_args["i"])

    ##Load saved RF model 
    ##assume that default SYMBOL for saved model is savedmodel
    trained_model = BSON.load(parsed_args["model_path"], @__MODULE__)[:currmodel]
    predictions = ScikitLearn.predict(trained_model, X)

    ##QC each variable in VARIALBES_TO_QC
    for var in VARIALBES_TO_QC

        ##Create new field to reshape QCed field to 
        NEW_FIELD = missings(Float64, cfrad_dims) 

        ##Only modify relevant data based on indexer, everything else should be fill value 
        QCED_FIELDS = input_cfrad[var][:][indexer]
        initial_count = count(!iszero, QCED_FIELDS)
        ##Apply predictions from model 
        ##If model predicts 1, this indicates a prediction of meteorological data 
        QCED_FIELDS = map(x -> Bool(predictions[x[1]]) ? x[2] : missing, enumerate(QCED_FIELDS))
        final_count = count(!ismissing, QCED_FIELDS)
        
        ###Need to reconstruct original 
        NEW_FIELD = NEW_FIELD[:]
        NEW_FIELD[indexer] = QCED_FIELDS
        NEW_FIELD = reshape(NEW_FIELD, cfrad_dims)

        defVar(input_cfrad, var * QC_SUFFIX, NEW_FIELD, ("range", "time"))

        println()
        printstyled("REMOVED $(initial_count - final_count) PRESUMED NON-METEORLOGICAL DATAPOINTS\n", color=:green)
        println("FINAL COUNT OF DATAPOINTS IN $(var): $(final_count)")
    end 
    println("DID IT WORK?")
end 

main()
