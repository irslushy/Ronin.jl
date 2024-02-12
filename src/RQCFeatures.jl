using Missings
using NCDatasets

center_weight::Float64 = 0

###Weight matrixes for calculating spatial parameters 
iso_weights::Matrix{Union{Missing, Float64}} = allowmissing(ones(7,7))
iso_weights[4,4] = center_weight 
iso_window::Tuple{Int64, Int64} = (7,7)

avg_weights::Matrix{Union{Missing, Float64}} = allowmissing(ones(5,5))
avg_weights[3,3] = center_weight 
avg_window::Tuple{Int64, Int64} = (5,5)

std_weights::Matrix{Union{Missing, Float64}} = allowmissing(ones(5,5))
std_weights[3,3] = center_weight 
std_window::Tuple{Int64, Int64} = (5,5)

EarthRadiusKm::Float64 = 6375.636
EarthRadiusSquar::Float64 = 6375.636 * 6375.636
DegToRad::Float64 = 0.01745329251994372

# Beamwidth of ELDORA and TDR
eff_beamwidth::Float64 = 1.8
beamwidth::Float64 = eff_beamwidth*0.017453292;

###Prefix of functions for calculating spatially averaged variables in the utils.jl file 
func_prefix::String= "calc_"
func_regex::Regex = r"(\w{1,})\((\w{1,})\)"

###List of functions currently implemented in the module
valid_funcs::Array{String} = ["AVG", "ISO", "STD", "AHT", "PGG", "RNG", "NRG"] 

FILL_VAL::Float64 = -32000.
RADAR_FILE_PREFIX::String = "cfrad"

##Threshold to exclude gates when gate >= value 
NCP_THRESHOLD::Float64 = .2 
PGG_THRESHOLD::Float64 = 1. 

REMOVE_LOW_NCP::Bool = true 
REMOVE_HIGH_PGG::Bool = true 


###Side note - I do realize that Julia will return the last statement of a function automatically, 
###but am including the return statement here for increased code clarity 

##Returns flattened version of NCP 
function get_NCP(data::NCDataset)
    ###Some ternary operator + short circuit trickery here 
    (("NCP" in keys(data)) ? (return(data["NCP"][:]))
                        : ("SQI" in keys(data) ||  error("Could Not Find NCP in dataset")))
    return(data["SQI"][:])
end

function get_RNG(data::NCDataset)
    return(repeat(data["range"][:], 1, length(data["time"])))
end 

function get_NRG(data::NCDataset)
    rngs = get_RNG(data)
    alts = repeat(transpose(data["altitude"][:]), length(data["range"]), 1)
    return(rngs ./ alts)
end 



function _weighted_func(var::AbstractMatrix{Union{Float32, Missing}}, weights::Matrix{Union{Missing, Float64}}, func)
        
    valid_weights = .!map(ismissing, weights)

    updated_weights = weights[valid_weights]
    updated_var = var[valid_weights]

    valid_idxs = .!map(ismissing, updated_var)

    ##Returns 0 when missing, 1 when not 
    return(func(updated_var[valid_idxs] .* updated_weights[valid_idxs]))
end

function _weighted_func(var, weights, func)
    valid_weights = .!map(ismissing, weights)

    updated_weights = weights[valid_weights]
    updated_var = var[valid_weights]

    valid_idxs = .!map(ismissing, updated_var)

    ##Returns 0 when missing, 1 when not 
    return(func(updated_var[valid_idxs] .* updated_weights[valid_idxs]))
end

function _weighted_func(var::Matrix{Float64}, weights::Matrix{Union{Missing, Float64}}, func)
    
    valid_weights = .!map(ismissing, weights)

    updated_weights = weights[valid_weights]
    updated_var = var[valid_weights]

    valid_idxs = .!map(ismissing, updated_var)

    ##Returns 0 when missing, 1 when not 
    result = func(updated_var[valid_idxs] .* updated_weights[valid_idxs])
    return result 
end


function calc_iso(var::AbstractMatrix{Union{Missing, Float64}};
    weights::Matrix{Union{Missing, Float64}} = iso_weights, 
    window::Matrix{Union{Missing, Float64}} = iso_window)

    # if size(weights) != window
    #     error("Weight matrix does not equal window size")
    # end

    missings = map((x) -> Float64(ismissing(x)), var)
    iso_array = mapwindow((x) -> _weighted_func(x, weights, sum), missings, window) 
end

##Calculate the isolation of a given variable 
###These actually don't necessarily need to be functions of their own, could just move them to
###Calls to _weighted_func 
function calc_iso(var::AbstractMatrix{Union{Missing, Float32}}; weights = iso_weights, window = iso_window)

    if size(weights) != window
        error("Weight matrix does not equal window size")
    end

    missings = map((x) -> Float64(ismissing(x)), var)
    iso_array = mapwindow((x) -> _weighted_func(x, weights, sum), missings, window) 
end

function airborne_ht(elevation_angle::Float64, antenna_range::Float64, aircraft_height::Float64)
    ##Initial heights are in meters, convert to km 
    aRange_km, acHeight_km = (antenna_range, aircraft_height) ./ 1000
    term1 = aRange_km^2 + EarthRadiusKm^2
    term2 = 2 * aRange_km * EarthRadiusKm * sin(deg2rad(elevation_angle))

    return sqrt(term1 + term2) - EarthRadiusKm + acHeight_km
end 


function prob_groundgate(elevation_angle, antenna_range, aircraft_height, azimuth)

    ###If range of gate is less than altitude, cannot hit ground
    ###If elevation angle is positive, cannot hit ground 
    if (antenna_range < aircraft_height || elevation_angle > 0)
        return 0. 
    end 
    elevation_rad, azimuth_rad = map((x) -> deg2rad(x), (elevation_angle, azimuth))
    
    #Range at which the beam will intersect the ground (Testud et. al 1995 or 1999?)
    Earth_Rad_M = EarthRadiusKm*1000
    ground_intersect = (-(aircraft_height)/sin(elevation_rad))*(1+aircraft_height/(2*Earth_Rad_M* (tan(elevation_rad)^2)))
    
    range_diff = ground_intersect - antenna_range
    
    if (range_diff <= 0)
            return 1.
    end 
    
    gelev = asin(-aircraft_height/antenna_range)
    elevation_offset = elevation_rad - gelev
    
    if elevation_offset < 0
        return 1.
    else
        
        beamaxis = sqrt(elevation_offset * elevation_offset)
        gprob = exp(-0.69314718055995*(beamaxis/beamwidth))
        
        if (gprob>1.0)
            return 1.
        else
            return gprob
        end
    end 
end 

##Calculate the windowed standard deviation of a given variablevariable 
function calc_std(var::AbstractMatrix{Union{Missing, Float64}}; weights = std_weights, window = std_window)
    if size(weights) != window
        error("Weight matrix does not equal window size")
    end

    mapwindow((x) -> _weighted_func(x, weights, std), var, window, border=Fill(missing))
end 

##Calculate the windowed standard deviation of a given variablevariable 
function calc_std(var::AbstractMatrix{}; weights = std_weights, window = std_window)
    if size(weights) != window
        error("Weight matrix does not equal window size")
    end

    mapwindow((x) -> _weighted_func(x, weights, std), var, window, border=Fill(missing))
end 

function calc_avg(var::Matrix{Union{Missing, Float32}}; weights = avg_weights, window = avg_window)

    if size(weights) != window
        error("Weight matrix does not equal window size")
    end

    mapwindow((x) -> _weighted_func(x, weights, mean), var, window, border=Fill(missing))
end

function calc_avg(var::Matrix{}; weights = avg_weights, window = avg_window)

    if size(weights) != window
        error("Weight matrix does not equal window size")
    end

    mapwindow((x) -> _weighted_func(x, weights, mean), var, window, border=Fill(missing))
end

function calc_pgg(cfrad::NCDataset)

    num_times = length(cfrad["time"])
    num_ranges = length(cfrad["range"])

    ranges = repeat(cfrad["range"][:], 1, num_times)
    ##Height, elevation, and azimuth will be the same for every ray
    heights = repeat(transpose(cfrad["altitude"][:]), num_ranges, 1)
    elevs = repeat(transpose(cfrad["elevation"][:]), num_ranges, 1)
    azimuths = repeat(transpose(cfrad["azimuth"][:]), num_ranges, 1)
    
    ##This would return 
    return(map((w,x,y,z) -> prob_groundgate(w,x,y,z), elevs, ranges, heights, azimuths))
end 

function calc_aht(cfrad::NCDataset)

    num_times = length(cfrad["time"])
    num_ranges = length(cfrad["range"])
    
    elevs = repeat(transpose(cfrad["elevation"][:]), num_ranges, 1)
    ranges = repeat(cfrad["range"][:], 1, num_times)
    heights = repeat(transpose(cfrad["altitude"][:]), num_ranges, 1)

    return(map((x,y,z) -> airborne_ht(Float64(x),Float64(y),Float64(z)), elevs, ranges, heights))

end 


####MOVE The below into RQCFeatures.jl 
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

###Parses the specified parameter file 
###Thinks of the parameter file as a list of tasks to perform on variables in the CFRAD
"Function to parse a given task list
Also performs checks to ensure that the specified 
tasks are able to be performed to the specified CFRad file"
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
                        printstyled("\"$token\" NOT FOUND IN CFRAD FILE.... POTENTIAL ERROR IN CONFIG FILE\n", color=:red)
                    end
                end
            end
        end 
    end 
    return(task_param_list)
end 

"""
Parses input parameter file for use in outputting feature names to 
    HDF5 file as attributes. NOTE: Cfradial-unaware. If one of the variables is 
    specified incorrectly in the parameter file, will cause errors
"""
function get_task_params(params_file; delimiter = ',')

    tasks = readlines(params_file)
    task_param_list = String[] 

    for line in tasks
        ###Ignore comments in the parameter file 
        if line[1] == '#'
            continue
        else
            delimited = split(line, delimiter)
            for token in delimited
                token = strip(token, ' ')
                expr_ret = match(func_regex,token)
                if (typeof(expr_ret) != Nothing)
                    if (expr_ret[1] ∉ valid_funcs)
                        println("ERROR: CANNOT CALCULATE $(expr_ret[1]) of $(expr_ret[2])\n", 
                        "Potentially invalid function or missing variable\n")
                    else
                        ###Add λ to the front of the token to indicate it is a function call
                        ###This helps later when trying to determine what to do with each "task" 
                        push!(task_param_list, token)
                    end 
                else 
                    push!(task_param_list, token)
                end 
            end 
        end 
    end 
    return task_param_list 
end 

"""
    Driver function that calculates a set of features from a single CFRadial file. Features are 
    specified in file located at argfile_path. If the file has already been manually QCed, set
    HAS_MANUAL_QC to true, and an additional Y array will be returned containing "1" if a datapoint
    was retained in manual QC and "0" if it was removed. Also included are flags to remove datapoints 
    with low Normalized Coherent Power (NCP) and high Probability of Ground (PGG). Finally, the 
    remove_variable argument specifies which field will be used to determine where "missing" data exists 
    and will subsequently be removed from the returned dataset. 

    Returns: 
        -X: Array that is dimensioned (num_gates x num_features) where num_gates is the number of valid 
            (non-missing, meeting NCP/PGG thresholds) the function finds, and num_features is the 
            number of features specified in the argument file to calculate. 

        -Y: IF HAS_MANUAL_QC == true, will return Y, array containing 1 if a datapoint was retained 
            during manual QC, and 0 otherwise. 

        -INDEXER: Based on remove_variable as described above, contains boolean array specifiying
                  where in the scan features valid data and where does not. 
"""
function process_single_file(cfrad::NCDataset, argfile_path; 
    HAS_MANUAL_QC = false, REMOVE_LOW_NCP = false, REMOVE_HIGH_PGG = false, remove_variable = "VV")

    cfrad_dims = (cfrad.dim["range"], cfrad.dim["time"])
    #println("\r\nDIMENSIONS: $(cfrad_dims[1]) times x $(cfrad_dims[2]) ranges\n")
    
    
    valid_vars = keys(cfrad)
    tasks = get_task_params(argfile_path, valid_vars)
    
    
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
            
            #println("CALCULATING $curr_func OF $var... ")
            #println("TYPE OF VARIABLE: $(typeof(cfrad[var][:,:]))")
            curr_func = Symbol(func_prefix * curr_func)
            startTime = time() 

            raw = @eval $curr_func($cfrad[$var][:,:])[:]
            filled = Vector{Float64}
            filled = [ismissing(x) || isnan(x) ? Float64(FILL_VAL) : Float64(x) for x in raw]

            any(isnan, filled) ? throw("NAN ERROR") : 

            X[:, i] = filled[:]
            calc_length = time() - startTime
            #println("Completed in $calc_length s"...)
            #println() 
            
        else 
            #println("GETTING: $task...")

            if (task == "PGG") 
                startTime = time() 
                ##Change missing values to FILL_VAL 
                PGG = [ismissing(x) || isnan(x) ? Float64(FILL_VAL) : Float64(x) for x in calc_pgg(cfrad)[:]]
                X[:, i] = PGG
                PGG_Completed_Flag = true 
                calc_length = time() - startTime
                #println("Completed in $calc_length s"...)
            elseif (task == "NCP")
                startTime = time()
                X[:, i] = [ismissing(x) || isnan(x) ? Float64(FILL_VAL) : Float64(x) for x in get_NCP(cfrad)[:]]
                calc_length = time() - startTime
                #println("Completed in $calc_length s"...)
            elseif (task == "AHT")
                startTime = time()
                X[:, i] = [ismissing(x) || isnan(x) ? Float64(FILL_VAL) : Float64(x) for x in calc_aht(cfrad)[:]]
                #println("Completed in $(time() - startTime) seconds")
            elseif (task == "RNG") 
                startTime = time() 
                X[:, i] = [ismissing(x) || isnan(x) ? Float64(FILL_VAL) : Float64(x) for x in get_RNG(cfrad)[:]]
                #println("Completed in $(time() - startTime) seconds")
            elseif (task == "NRG")
                startTime = time()
                X[:, i] = [ismissing(x) || isnan(x) ? Float64(FILL_VAL) : Float64(x) for x in get_NRG(cfrad)[:]]
                #println("Completed in $(time() - startTime) seconds")
            else
                startTime = time() 
                X[:, i] = [ismissing(x) || isnan(x) ? Float64(FILL_VAL) : Float64(x) for x in cfrad[task][:]]
                calc_length = time() - startTime
                #println("Completed in $calc_length s"...)
            end 
            #println()
        end 
    end


    ###Uses INDEXER to remove data not meeting basic quality thresholds
    ###A value of 0 in INDEXER will remove the data from training/evaluation 
    ###by the subsequent random forest model 

    ###Begin by simply removing the gates where no velocity is found 
    #println("REMOVING MISSING DATA BASED ON $(remove_variable)...")
    starttime = time() 
    VT = cfrad[remove_variable][:]

    ###Indexer will be boolean array determining whether or not to
    ###retain a gate for output into the model training/testing set 
    INDEXER = [ismissing(x) ? false : true for x in VT]
    #println("COMPLETED IN $(round(time()-starttime, sigdigits=4))s")
    #println("") 

    #println("FILTERING")
    starttime=time()

    if (REMOVE_LOW_NCP)
        #println("REMOVING BASED ON NCP")
        #println("INITIAL COUNT: $(count(INDEXER))")
        NCP = get_NCP(cfrad)
        ###Only need to modify the portions of the indexer that are currently true
        INDEXER[INDEXER] = [x <= NCP_THRESHOLD ? false : true for x in NCP[INDEXER]]
        #println("FINAL COUNT: $(count(INDEXER))")
    end

    if (REMOVE_HIGH_PGG)

        #println("REMOVING BASED ON PGG")
        #println("INITIAL COUNT: $(count(INDEXER))")
        
        if (PGG_Completed_Flag)
            INDEXER[INDEXER] = [x >= PGG_THRESHOLD ? false : true for x in PGG[INDEXER]]
        else
            PGG = [ismissing(x) || isnan(x) ? Float64(FILL_VAL) : Float64(x) for x in calc_pgg(cfrad)[:]]
            INDEXER[INDEXER] = [x >= PGG_THRESHOLD ? false : true for x in PGG[INDEXER]]
        end

        #println("FINAL COUNT: $(count(INDEXER))")
        #println()
    end
    
    #println("COMPLETED IN $(round(time()-starttime, sigdigits=4))s")


    #println("INDEXER SHAPE: $(size(INDEXER))")
    #println("X SHAPE: $(size(X))")

    X = X[INDEXER, :] 
    #println("NEW X SHAPE: $(size(X))")
    
    ###Allows for use with already QC'ed files to output a Y array for 
    ###model training 
    if HAS_MANUAL_QC

        #println("Parsing METEOROLOGICAL/NON METEOROLOGICAL data")
        startTime = time() 
        ###try catch block here to see if the scan has manual QC
        ###Filter the input arrays first 
        VG = cfrad["VG"][:][INDEXER]
        VV = cfrad["VV"][:][INDEXER]

        Y = reshape([ismissing(x) ? 0 : 1 for x in VG .- VV][:], (:, 1))
        calc_length = time() - startTime
        #println("Completed in $calc_length s"...)
        #println()


        #println()
        #println("FINAL X SHAPE: $(size(X))")
        #println("FINAL Y SHAPE: $(size(Y))")

        return(X, Y, INDEXER)
    else
        #println("NO MANUAL QC")
        #println("FINAL X SHAPE: $(size(X))")
        return(X, false, INDEXER)
    end 
end 
    
    ##Applies function given by func to the weighted version of the matrix given by var 
    ##Also applies controls for missing variables to both weights and var 
    ##If there are Missing values in either the variable or the weights, they will be ignored 
    
    #precompile(_weighted_func, (AbstractMatrix{}, Matrix{}))


