include("./src/JMLQC_utils.jl")
using .JMLQC_utils 
using ArgParse 
using HDF5 

###Set up argument table for CLI 
function parse_commandline()
    
    s = ArgParseSettings()

    @add_arg_table s begin
        
        "training_data_path"
            help = "Path to input training data h5"
            required = true
        "training_data_out"
            help = ("Location to output training data minus validation set")
            required = true 
        "validation_data_out"
            help = "Location to output validation data to"
            required = true
    end

    return parse_args(s)
end


function main() 
    args = parse_commandline()
    print(args["training_data_path"])
    JMLQC_utils.remove_validation(args["training_data_path"]; training_output = args["training_data_out"], validation_output = args["validation_data_out"])
end 

main()