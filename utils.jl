using NCDatasets
using ImageFiltering
using Statistics
using Images
using Missings
using BenchmarkTools

EarthRadiusKm = 6375.636
EarthRadiusSquare = 6375.636 * 6375.636
DegToRad = 0.01745329251994372

# Beamwidth of ELDORA and TDR
eff_beamwidth = 1.8
beamwidth = eff_beamwidth*0.017453292;


##Weight matrixes for calculating spatial variables 
##TODO: Add these as default arguments for their respective functions 

USE_GATE_IN_CALC = false 

function get_NCP(data::NCDataset)
    ###Some ternary operator + short circuit trickery here 
    (("NCP" in keys(data)) ? (return(data["NCP"][:,start_scan_idx:1:end_scan_idx]))
                        : ("SQI" in keys(data) ||  error("Could Not Find NCP in dataset")))
    return(data["SQI"][:, start_scan_idx:1:end_scan_idx])
end

##Applies function given by func to the weighted version of the matrix given by var 
##Also applies controls for missing variables to both weights and var 
function _weighted_func(var::AbstractMatrix{}, weights, func)
    
    valid_weights = .!map(ismissing, weights)
    
    
    updated_weights = weights[valid_weights]
    updated_var = var[valid_weights]
    
    valid_idxs = .!map(ismissing, updated_var)
    
    ##Returns 0 when missing, 1 when not 
    return func(updated_var[valid_idxs] .* updated_weights[valid_idxs])
end

function calc_isolation_param(var::AbstractMatrix{Union{Missing, Float64}}, weights, window)
    
    if size(weights) != window
        error("Weight matrix does not equal window size")
    end
    
    missings = map((x) -> ismissing(x), var)
    iso_array = mapwindow((x) -> _weighted_func(x, weights, sum), missings, window) 
end

##Calculate the isolation of a given variable 
###These actually don't necessarily need to be functions of their own, could just move them to
###Calls to _weighted_func 
function calc_isolation_param(var::AbstractMatrix{Union{Missing, Float32}}, weights, window)
    
    if size(weights) != window
        error("Weight matrix does not equal window size")
    end
    
    missings = map((x) -> ismissing(x), var)
    iso_array = mapwindow((x) -> _weighted_func(x, weights, sum), missings, window) 
end

###Emulates np.nanmean function while implementing weighted averaging

function_missing_weights_avg(var::AbstractMatrix{Union{Missing, Float32}}, weights::AbstractMatrix{Union{Missing, Float64}})

    valid_weights = .!map(ismissing, weights)
    
    
    updated_weights = weights[valid_weights]
    updated_var = var[valid_weights]
    
    valid_idxs = .!map(ismissing, updated_var)
    return(mean((updated_var[valid_idxs] .* updated_weights[valid_idxs])))
end

function_missing_weights_avg(var::AbstractMatrix{Union{Missing, Float64}}, weights::AbstractMatrix{Union{Missing, Float64}})

    valid_weights = .!map(ismissing, weights)
    
    
    updated_weights = weights[valid_weights]
    updated_var = var[valid_weights]
    
    valid_idxs = .!map(ismissing, updated_var)
    return(mean((updated_var[valid_idxs] .* updated_weights[valid_idxs])))
end

function _missing_weights_avg(var, weights) 
    
    valid_weights = .!map(ismissing, weights)
    
    
    updated_weights = weights[valid_weights]
    updated_var = var[valid_weights]
    
    valid_idxs = .!map(ismissing, updated_var)
    return(mean((updated_var[valid_idxs] .* updated_weights[valid_idxs])))
end

function airborne_ht(elevation_angle, antenna_range, aircraft_height)
#     global GLOBL_COUNTER
#     print(GLOBL_COUNTER)
#     GLOBL_COUNTER = GLOBL_COUNTER + 1
    ##Initial heights are in meters, convert to km 
    aRange_km, acHeight_km = (antenna_range, aircraft_height) ./ 1000
    term1 = aRange_km^2 + EarthRadiusKm^2
    term2 = 2 * aRange_km * EarthRadiusKm * sin(deg2rad(elevation_angle))
        
    return sqrt(term1 + term2) - EarthRadiusKm + acHeight_km
end 

###How can we optimize this.... if one gate has a probability of ground equal to 1, the rest of 
###The gates further in that range must also have probability equal to 1, correct? 
function prob_groundgate(elevation_angle, antenna_range, aircraft_height, azimuth) #max_range) ? ?? ? ?? ? ?? 
    
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
