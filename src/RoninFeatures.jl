using Missings
using NCDatasets
using Base.Threads 
include("./RoninConstants.jl")


function missing_std(data)

    if count(ismissing, data) == length(data)
        return FILL_VAL 
    end 
    
    std(skipmissing(data))
end 

function missing_avg(data)

    if count(ismissing, data) == length(data)
        return FILL_VAL  
    end 

    mean(skipmissing(data))
end 


##Returns flattened version of NCP 
function calc_ncp(data::NCDataset)
    ###Some ternary operator + short circuit trickery here 
    (("NCP" in keys(data)) ? (return(data["NCP"][:]))
                        : ("SQI" in keys(data) ||  error("Could Not Find NCP in dataset")))
    return(data["SQI"][:])
end

function calc_rng(data::NCDataset)
    return(repeat(data["range"][:], 1, length(data["time"])))
end 

function calc_nrg(data::NCDataset)
    rngs = calc_rng(data)
    alts = repeat(transpose(data["altitude"][:]), length(data["range"]), 1)
    return(rngs ./ alts)
end 



function _weighted_func(var::AbstractMatrix{Union{Float32, Missing}}, weights::Matrix{Union{Missing, Float32}}, func)
    ##Returns 0 when missing, 1 when not 
    return(func(var .* weights))
end

function _weighted_func(var, weights, func)
    return(func(var .* weights))
end

function _weighted_func(var::Matrix{Float32}, weights::Matrix{Union{Missing, Float32}}, func)
    return(func(var .* weights))
end


function calc_iso(var::AbstractMatrix{Union{Missing, Float32}};
    weights::Matrix{Union{Missing, Float32}} = iso_weights, 
    window = iso_window)

    ###Where is missing? 
    missings = map((x) -> Float32(ismissing(x)), var)
    ###Calculate sum of missing gates in the windowed area for each gate in scan 
    iso_array = mapwindow((x) -> _weighted_func(x, weights, sum), missings, window) 
end

##Calculate the isolation of a given variable 
###These actually don't necessarily need to be functions of their own, could just move them to
###Calls to _weighted_func 
function calc_iso(var::AbstractMatrix{Union{Missing, Float32}}; weights = iso_weights, window = iso_window)
    missings = map((x) -> Float32(ismissing(x)), var)
    iso_array = mapwindow((x) -> _weighted_func(x, weights, sum), missings, window) 
end

function airborne_ht(elevation_angle::Float32, antenna_range::Float32, aircraft_height::Float32)
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
function calc_std(var::AbstractMatrix{Union{Missing, Float32}}; weights = std_weights, window = std_window)
    mapwindow((x) -> _weighted_func(x, weights, missing_std), var, window, border=Fill(missing))
end 

##Calculate the windowed standard deviation of a given variablevariable 

function calc_std(var::AbstractMatrix{}; weights = std_weights, window = std_window)
    global REPLACE_MISSING_WITH_FILL

    if ( REPLACE_MISSING_WITH_FILL)
        @inbounds var[map(ismissing, var)] .= FILL_VAL
    end 

    @inbounds mapwindow((x) -> _weighted_func(x, weights, missing_std), var, window, border=Fill(missing))
end 

function calc_avg(var::Matrix{Union{Missing, Float32}}; weights = avg_weights, window = avg_window)

    if ( REPLACE_MISSING_WITH_FILL)
        @inbounds var[map(ismissing, var)] .= FILL_VAL
    end 

    @inbounds mapwindow((x) -> _weighted_func(x, weights, missing_avg), var, window, border=Fill(missing))
end

function calc_avg(var::Matrix{}; weights = avg_weights, window = avg_window)

    if (REPLACE_MISSING_WITH_FILL)
        @inbounds var[map(ismissing, var)] .= FILL_VAL
    end 

    @inbounds mapwindow((x) -> _weighted_func(x, weights, missing_avg), var, window, border=Fill(missing))
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
    @inbounds return(map((w,x,y,z) -> prob_groundgate(w,x,y,z), elevs, ranges, heights, azimuths))
end 

function calc_aht(cfrad::NCDataset)

    num_times = length(cfrad["time"])
    num_ranges = length(cfrad["range"])
    
    elevs = repeat(transpose(cfrad["elevation"][:]), num_ranges, 1)
    ranges = repeat(cfrad["range"][:], 1, num_times)
    heights = repeat(transpose(cfrad["altitude"][:]), num_ranges, 1)

    @inbounds return(map((x,y,z) -> airborne_ht(Float32(x),Float32(y),Float32(z)), elevs, ranges, heights))

end 


####MOVE The below into RQCFeatures.jl 
function get_num_tasks(params_file; delimeter = ",")

    tasks = readlines(params_file)
    num_tasks = 0 

    for line in tasks
        if line[1] == '#'
            continue
        else 
            delimited = split(strip(line), delimeter) 
            for token in delimited
                if !(token == "")
                    num_tasks += 1
                end 
            end 
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
    

    full_tasks = ""
    for itm in tasks
        if (itm != "") && (itm[1] != "#")
            if full_tasks == ""
                full_tasks = strip(itm, ',')
            else 
                full_tasks = full_tasks * "," * strip(itm, ',')
            end 
        end 
    end 

    
    delimited = split(full_tasks, delimiter)
    for token in delimited
        ###Remove whitespace and see if the token is of the form ABC(DEF)
        token = strip(token, ' ')
        expr_ret = match(func_regex,token)

        if token == ""
            printstyled("WARNING: Empty Token in config file... continuing to next token...\n", color=:yellow) 
            continue 
        end 
        ###If it is, make sure that it is both a valid function and a valid variable 
        if (typeof(expr_ret) != Nothing)
            if (expr_ret[1] ∉ valid_funcs || 
                (expr_ret[2] ∉ variablelist && expr_ret[2] ∉ valid_derived_params))
                println("ERROR: CANNOT CALCULATE $(expr_ret[1]) of $(expr_ret[2])\n", 
                "Potentially invalid function or missing variable\n")
            else
                ###Add λ to the front of the token to indicate it is a function call
                ###This helps later when trying to determine what to do with each "task" 
                push!(task_param_list, token)
            end 
        else
            ###Otherwise, check to see if this is a valid variable 
            if token in variablelist || token ∈ valid_funcs || token ∈ valid_derived_params 
                push!(task_param_list, token)
            else
                printstyled("\"$token\" NOT FOUND IN CFRAD FILE.... POTENTIAL ERROR IN CONFIG FILE\n", color=:red)
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

    full_tasks = ""
    for itm in tasks
        if (itm != "") && (itm[1] != "#")
            if full_tasks == ""
                full_tasks = strip(itm, ',')
            else 
                full_tasks = full_tasks * "," * strip(itm, ',')
            end 
        end 
    end 



    delimited = split(strip(full_tasks), delimiter)
    for token in delimited
        token = strip(token, ' ')
        if token == ""
            continue 
        end 
        expr_ret = match(func_regex,token)

        if (typeof(expr_ret) != Nothing)
            if (expr_ret[1] ∉ valid_funcs)
                println("ERROR: CANNOT CALCULATE $(expr_ret[1]) of $(expr_ret[2])\n", 
                "Potentially invalid function or missing variable\n")
            else
                push!(task_param_list, token)
            end 
        else 
            push!(task_param_list, token)                 
        end 
    end 

    return task_param_list 
end 

"""

Driver function that calculates a set of features from a single CFRadial file. Features are 
specified in file located at argfile_path. 

Will return a tuple of (X, Y, indexer) where X is the features matrix,  `Y`, a matrix containing the verification 
- where human QC determined the gate was meteorological (value of 1), or non-meteorological (value of 0), 
and `indexer` contains a vector of booleans describing which gates met basic quality control thresholds 
and thus are represented in the X and Y matrixes

Weight matrixes are specified in file header, or passed as explicit argument. 

# Required arguments 

```julia 
cfrad::NCDataset 
```
Input `NCDataset` containing radar scan variables 

```julia 
argfile_path::String 
```
Path to file contaning config/task information. 

# Optional keyword arguments 

```julia 
HAS_INTERACTIVE_QC::Bool = false
```
If the scan has already had a human apply quality control to it, set to `true`. Otherwise, `false`

```julia 
REMOVE_LOW_NCP::Bool = false
```
Whether or not to ignore gates that do not meet a minimum NCP/SQI threshold. If `true`, these gates will be
set to `false` in `indexer`, and features/verification will not be calculated for them. 

```julia 
NCP_THRESHOLD::Float32 = .2
```
Theshold at or below which to remove data 

```julia 
REMOVE_HIGH_PGG::Bool = false
```
Whether or not to ignore gates that exceed a given Probability of Ground Gate(PGG) threshold. If `true`, these gates will be
set to `false` in `indexer`, and features/verification will not be calculated for them. 

```julia
PGG_THRESHOLD
```
Threshold at or above which to remove data 

```julia 
QC_variable::String = "VG"
```
Name of a variable in input CFRadial file that has had QC applied to it already. Used to calculate verification `Y` matrix. 

```julia 
remove_variable::String = "VV" 
```
Name of raw variable in input CFRadial file that will be used to determine where `missing` gates exist in the sweep.

```julia 
replace_missing::Bool = false
```
For spatial parameters, whether or not to replace `missings` values with `FILL_VAL`

--- 

# Returns: 

    -X: Array that is dimensioned (num_gates x num_features) where num_gates is the number of valid 
        (non-missing, meeting NCP/PGG thresholds) the function finds, and num_features is the 
        number of features specified in the argument file to calculate. 

    -Y: IF HAS_INTERACTIVE_QC == true, will return Y, array containing 1 if a datapoint was retained 
        during interactive QC, and 0 otherwise. 

    -INDEXER: Based on remove_variable as described above, contains boolean array specifiying
                where in the scan features valid data and where does not. 
"""
function process_single_file(cfrad::NCDataset, argfile_path::String; 
    HAS_INTERACTIVE_QC::Bool = false, REMOVE_LOW_NCP::Bool = false, NCP_THRESHOLD::Float32 = .2, REMOVE_HIGH_PGG::Bool = false,
    PGG_THRESHOLD::Float32 = 1.,  QC_variable::String = "VG", remove_variable::String = "VV", replace_missing::Bool=false,
    mask_features::Bool = false, feature_mask::Matrix{Bool} = [true true ; false false;], 
        weight_matrixes::Vector{Matrix{Union{Missing, Float32}}}= [Matrix{Union{Missing, Float32}}(undef, 0,0)])

    cfrad_dims = (cfrad.dim["range"], cfrad.dim["time"])
    #println("\r\nDIMENSIONS: $(cfrad_dims[1]) times x $(cfrad_dims[2]) ranges\n")
    
    if replace_missing
        global REPLACE_MISSING_WITH_FILL = true 
    else 
        global REPLACE_MISSING_WITH_FILL = false 
    end 

    valid_vars = keys(cfrad)
    tasks = get_task_params(argfile_path, valid_vars)

    
    ###Features array 
    X = Matrix{Float32}(undef,cfrad.dim["time"] * cfrad.dim["range"], length(tasks))

    ###Array to hold PGG for indexing  
    PGG = Matrix{Float32}(undef, cfrad.dim["time"]*cfrad.dim["range"], 1)

    PGG_Completed_Flag = false 
    NCP_Completed_Flag = false 
    add_weight_matrix = (weight_matrixes != [Matrix{Union{Missing, Float32}}(undef, 0,0)]) & (length(weight_matrixes) == length(tasks))
    ### 
    if (! add_weight_matrix) & (weight_matrixes != [Matrix{Union{Missing, Float32}}(undef, 0,0)])
        print("WEIGHT MATRIXES: $(weight_matrixes)")
        throw("ERROR: Weight matrixes are not default but do not match length of tasks... terminating...")
    end 
    
    for (i, task) in enumerate(tasks)

        ###Test to see if task involves a spatial parameter function call) 
        ###A match implies spatial parameter, otherwise we assume it is just 
        ###simple derived parameter.

        ###Spatial parameters have general structure of calc_prm(cfrad_field)
        ###Where PRM is the 3 letter abbreviation of the spatial paramter (AVG, STD, etc.)
        ###and the cfrad_field is the variable we are performing this on 

        ###Non-spatial parameters have general structure of calc_prm(cfrad)
        ###Where PRM is the 3 lettter abbreviation of the parameter
        ###and the full cfradial is passed because oftentimes multiple fields are requried 
        regex_match = match(func_regex, task) 
        

        ###The only case that will need to be modified outside of the end indexer is the spatial parameters
        if (!isnothing(regex_match))

            startTime = time() 

            func = Symbol(func_prefix * lowercase(regex_match[1]))
            var = regex_match[2]
            
            
            if add_weight_matrix
                weight_matrix = weight_matrixes[i] 
                window_size = size(weight_matrix)
            else 
                currname = Symbol(lowercase(regex_match[1]) * "_weights")
                weight_matrix = @eval($currname)
                window_size = size(weight_matrix)
            end 


            if mask_features
                currdat = cfrad[var][:,:]
                ##Set invalid data to missing for spatial parameter calculations. This way, they 
                ##will be ignored. 
                currdat[.! feature_mask] .= missing 
                raw = @eval $func($currdat)[:]
            else 
                raw = @eval $func($cfrad[$var][:,:]; weights=$weight_matrix, window = $window_size )[:]
            end 

            filled = [ismissing(x) || isnan(x) ? Float32(FILL_VAL) : Float32(x) for x in raw]
        
            any(isnan, filled) ? throw("NAN ERROR") : 

            X[:, i] = filled[:]
            calc_length = time() - startTime

        ###For any of the other tasks, we don't need to mask the features beforehand because 
        ###They do not have any spatial dependence on missing gates - they are only a gatewise function 
        ###IF a new function is implemented that DOES have spatial context, that will need to be changed
        elseif (task in valid_derived_params)

            startTime = time() 

            func = Symbol(func_prefix * lowercase(task))
            raw = @eval $func($cfrad)[:]
            X[:, i] = [ismissing(x) || isnan(x) ? Float32(FILL_VAL) : Float32(x) for x in raw]

            if (task == "PGG")
                PGG_Completed_Flag = true
                PGG = X[:, i]
            end 

            if (task == "NCP" || task == "SQI") 
                NCP_Completed_Flag = true 
                NCP = X[:, i]
            end 

            calc_length = time() - startTime 
        
        ###Otherwise it's just a variable from the cfrad 
        else 
            startTime = time() 
            X[:, i] = [ismissing(x) || isnan(x) ? Float32(FILL_VAL) : Float32(x) for x in cfrad[task][:]]
            calc_length = time() - startTime
        end 
    end


    ###Uses INDEXER to remove data not meeting basic quality thresholds
    ###A value of 0/False in INDEXER will remove the data from training/evaluation 
    ###by the subsequent random forest model 

    ###Begin by simply removing the gates where no velocity is found 

    starttime = time() 

    VT = cfrad[remove_variable][:]
    INDEXER = [ismissing(x) for x in VT]
    ##Initial gates to focus on 
  
    #If there is missing data for the indexer variable at the current gate, 
    #Get rid of it, otherwise go with the original value in the flat mask 
    if mask_features 
        ###This is a little confusing, but basically currently the INDEXER contains whether or not 
        ###the value is missing in the original scan. If so, we can already say it's not a valid gate.
        ###If non-missing, just do whatever the QC_mask says 
        INDEXER = [INDEXER[i] ? false : maskval for (i, maskval) in enumerate(feature_mask[:])]
    else 
        INDEXER = .! INDEXER
    end 

    ###This got a little confusing so adding a comment here. 
    ###INDEXER now contains a value of true for valid indices (meeting basic QC thresholds + non missing)
    ###and false for invaldi indicies.  
    
    starttime=time()

    ###Do we need a check for where stuff is equal fill val here? 
    if (REMOVE_LOW_NCP)
        ###Remove data that does not equal or exceed minimum NCP threshold 
        ###Only need to do this for indicies that are true in INDEXER, as this is only remaining valid data
        if (NCP_Completed_Flag) 
            INDEXER[INDEXER] = [x <= NCP_THRESHOLD ? false : true for x in NCP[INDEXER]]
        else 
            NCP = [ismissing(x) || isnan(x) ? Float32(FILL_VAL) : Float32(x) for x in calc_ncp(cfrad)[:]]
            INDEXER[INDEXER] = [ x <= NCP_THRESHOLD ? false : true for x in NCP[INDEXER]]
        end 
    end

    if (REMOVE_HIGH_PGG)
        
        ###Remove data that is not equal to or less than maximum probability of ground gate 
        ###Only need to do this for indicies that are true in INDEXER, as this is only remaining valid data
        if (PGG_Completed_Flag)
            INDEXER[INDEXER] = [x >= PGG_THRESHOLD ? false : true for x in PGG[INDEXER]]
        else
            PGG = [ismissing(x) || isnan(x) ? Float32(FILL_VAL) : Float32(x) for x in calc_pgg(cfrad)[:]]
            INDEXER[INDEXER] = [x >= PGG_THRESHOLD ? false : true for x in PGG[INDEXER]]
        end

    end

    ###Subset the returned features to only the values that have cleared QC thresholds 
    X = X[INDEXER, :] 

    
    ###Allows for use with already QC'ed files to output a Y array for 
    ###model training 
    if HAS_INTERACTIVE_QC

        #println("Parsing METEOROLOGICAL/NON METEOROLOGICAL data")
        startTime = time() 
        ###try catch block here to see if the scan has interactive QC
        ###Filter the input arrays first 
        VG = cfrad[QC_variable][:][INDEXER]
        VV = cfrad[remove_variable][:][INDEXER]

        Y = reshape([ismissing(x) ? 0 : 1 for x in VG .- VV][:], (:, 1))
        calc_length = time() - startTime

        return(X, Y, INDEXER)
    else

        return(X, false, INDEXER)
    end 
end 




###Multiple dispatch version of process_single_file that allows user to specify a vector of weight matrixes 
###In this case will also pass the tasks to complete as a vector 
###weight_matrixes are also implicitly the window size 
function process_single_file(cfrad::NCDataset, tasks::Vector{String}, weight_matrixes::Vector{Matrix{Union{Missing, Float32}}}; 
    HAS_INTERACTIVE_QC = false, REMOVE_LOW_NCP = false, REMOVE_HIGH_PGG = false, QC_variable::String = "VG", remove_variable::String = "VV",
    replace_missing::Bool = false, mask_features::Bool=false, feature_mask::Matrix{Bool} = [true true ; false false])


    global REPLACE_MISSING_WITH_FILL

    if replace_missing
        global REPLACE_MISSING_WITH_FILL = true 
    else 
        global REPLACE_MISSING_WITH_FILL = false 
    end 

    valid_vars = keys(cfrad)

    ###Features array 
    X = Matrix{Float32}(undef,cfrad.dim["time"] * cfrad.dim["range"], length(tasks))

    ###Array to hold PGG for indexing  
    PGG = Matrix{Float32}(undef, cfrad.dim["time"]*cfrad.dim["range"], 1)

    PGG_Completed_Flag = false 
    NCP_Completed_Flag = false 

    for (i, task) in enumerate(tasks)

        ###Test to see if task involves a spatial parameter function call) 
        ###A match implies spatial parameter, otherwise we assume it is just 
        ###simple derived parameter.

        ###Spatial parameters have general structure of calc_prm(cfrad_field)
        ###Where PRM is the 3 letter abbreviation of the spatial paramter (AVG, STD, etc.)
        ###and the cfrad_field is the variable we are performing this on 

        ###Non-spatial parameters have general structure of calc_prm(cfrad)
        ###Where PRM is the 3 lettter abbreviation of the parameter
        ###and the full cfradial is passed because oftentimes multiple fields are requried 
        regex_match = match(func_regex, task) 
        
        
        ###If we get a match, we have a spatial parameter function 
        ###Weights are only important for these values 
        if (!isnothing(regex_match))

            startTime = time() 

            func = Symbol(func_prefix * lowercase(regex_match[1]))
            var = regex_match[2]

            weight_matrix = weight_matrixes[i]
            window_size = size(weight_matrix)
           
            if mask_features
                currdat = cfrad[var][:,:]
                currdat[.! feature_mask] .= missing 
                raw = @eval $func($currdat)[:]
            else 
                raw = @eval $func($cfrad[$var][:,:]; weights = $weight_matrix, window = $window_size)[:]
            end 

            filled = [ismissing(x) || isnan(x) ? Float32(FILL_VAL) : Float32(x) for x in raw]
            
            any(isnan, filled) ? throw("NAN ERROR") : 

            X[:, i] = filled[:]
            calc_length = time() - startTime


        elseif (task in valid_derived_params)

            startTime = time() 

            func = Symbol(func_prefix * lowercase(task))
            raw = @eval $func($cfrad)[:]
            X[:, i] = [ismissing(x) || isnan(x) ? Float32(FILL_VAL) : Float32(x) for x in raw]

            if (task == "PGG")
                PGG_Completed_Flag = true
                PGG = X[:, i]
            end 

            if (task == "NCP" || task == "SQI") 
                NCP_Completed_Flag = true 
                NCP = X[:, i]
            end 

            calc_length = time() - startTime 
        
        ###Otherwise it's just a variable from the cfrad 
        else 
            startTime = time() 
            X[:, i] = [ismissing(x) || isnan(x) ? Float32(FILL_VAL) : Float32(x) for x in cfrad[task][:]]
            calc_length = time() - startTime
        end 
    end


    ###Uses INDEXER to remove data not meeting basic quality thresholds
    ###A value of 0 in INDEXER will remove the data from training/evaluation 
    ###by the subsequent random forest model 

    ###Begin by simply removing the gates where no velocity is found 
    #println("REMOVING MISSING DATA BASED ON $(remove_variable)...")
    starttime = time() 

    VT = cfrad[remove_variable][:]
    INDEXER = [ismissing(x) for x in VT]
    
    ###After this, INDEXER will contain a value of TRUE if the gate is NON-MISSING and VALID in the feature mask, 
    ###FALSE otherwise 
    if mask_features 
        INDEXER = [INDEXER[i] ? false : maskval for (i, maskval) in enumerate(feature_mask[:]) ]
    else 
        INDEXER = .! INDEXER 
    end 
    
    starttime=time()

    if (REMOVE_LOW_NCP)
        ##Calculate NCP and set values that are below the threshold to invalid in the indexer 
        if (NCP_Completed_Flag) 
            INDEXER[INDEXER] = [x <= NCP_THRESHOLD ? false : true for x in NCP[INDEXER]]
        else 
            NCP = [ismissing(x) || isnan(x) ? Float32(FILL_VAL) : Float32(x) for x in calc_ncp(cfrad)[:]]
            INDEXER[INDEXER] = [ x <= NCP_THRESHOLD ? false : true for x in NCP[INDEXER]]
        end 
    end

    if (REMOVE_HIGH_PGG)
        ##Calculate PGG and set values that are above the threshold to invalid 
        if (PGG_Completed_Flag)
            INDEXER[INDEXER] = [x >= PGG_THRESHOLD ? false : true for x in PGG[INDEXER]]
        else
            PGG = [ismissing(x) || isnan(x) ? Float32(FILL_VAL) : Float32(x) for x in calc_pgg(cfrad)[:]]
            INDEXER[INDEXER] = [((x >= PGG_THRESHOLD) .& (x != FILL_VAL)) ? false : true for x in PGG[INDEXER]]
        end

    end
    
    X = X[INDEXER, :] 

    
    ###Allows for use with already QC'ed files to output a Y array for 
    ###model training 
    if HAS_INTERACTIVE_QC

        #println("Parsing METEOROLOGICAL/NON METEOROLOGICAL data")
        startTime = time() 
        ###try catch block here to see if the scan has interactive QC
        ###Filter the input arrays first 
        VG = cfrad[QC_variable][:][INDEXER]
        VV = cfrad[remove_variable][:][INDEXER]

        Y = reshape([ismissing(x) ? 0 : 1 for x in VG .- VV][:], (:, 1))
        calc_length = time() - startTime

        return(X, Y, INDEXER)
    else
        return(X, false, INDEXER)
    end 
end 


end 




