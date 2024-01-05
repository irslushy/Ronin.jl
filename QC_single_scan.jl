using NCDatasets
using HDF5
using MLJ
using PyCall, PyCallUtils, BSON
using ArgParse

include("./utils.jl")
using .JMLQC_utils 

using ScikitLearn
@sk_import ensemble: RandomForestClassifier

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
        "--outfile", "-o"
            help = "Location to output mined data to"
            default = "./mined_data.h5"

        
    end

    return parse_args(s)
end

function main()     
    ##Could make this a try-catch block if you really wanted to be 
    ##Extra safe about it 
    parsed_args = parse_commandline() 

    input_cfrad = NCDataset(parsed_args["CFRad_path"])
    cfrad_dims = (input_cfrad.dim["range"], input_cfrad.dim["time"])

    ###Will generally NOT return Y, but only (X, indexer)
    X, Y, indexer = JMLQC_utils.process_single_file(input_cfrad, keys(input_cfrad), parsed_args)

    ##Load saved RF model 
    ##assume that default SYMBOL for saved model is savedmodel
    trained_model = BSON.load(parsed_args["model_path"], @__MODULE__)[:currmodel]
    predictions = ScikitLearn.predict(trained_model, X)

    ##Create new field to reshape QCed field to 
    NEW_FIELD = Matrix{Float64}(undef, cfrad_dims)

    ##Parse relevant DBZ data by applying indexer 
    QCED_FIELDS = input_cfrad["ZZ"][:][indexer]
    initial_count = count(!iszero, QCED_FIELDS)
    ##Apply predictions from model 
    ##If model predicts 1, this indicates a prediction of meteorological data 
    ##WHY are we still having to deal with missings here? 
    QCED_FIELDS = map(x -> Bool(predictions[x[1]]) ? x[2] : 0, enumerate(QCED_FIELDS))
    final_count = count(!iszero, QCED_FIELDS)
    
    println()
    printstyled("INITIAL COUNT OF VALID DATAPOINTS: $(initial_count)
    \nFINAL COUNT: $(final_count)\n", color=:blue)
    ##TOO: Add fields to QC to parameter file 
    #printstyled("ACCURACY: $(round(sum(predictions .== Y) / length(Y) * 100, sigdigits=3)) %\n", color=:green)

end 

main()
