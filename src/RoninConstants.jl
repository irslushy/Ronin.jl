

center_weight::Float64 = 0

iso_weights::Matrix{Union{Missing, Float64}} = allowmissing(ones(7,7))
iso_weights[4,4] = center_weight 
iso_window::Tuple{Int64, Int64} = (7,7)

avg_weights::Matrix{Union{Missing, Float64}} = allowmissing(ones(5,5))
avg_weights[3,3] = center_weight 
avg_window::Tuple{Int64, Int64} = (5,5)

std_weights::Matrix{Union{Missing, Float64}} = allowmissing(ones(5,5))
std_weights[3,3] = center_weight 
std_window::Tuple{Int64, Int64} = (5,5)

vertical_window = Matrix{Union{Missing, Float64}}(missings(5,5))
vertical_window[:, 3] .= 1.

horiz_window = Matrix{Union{Missing, Float64}}(missings(5,5))
horiz_window[3, :] .= 1.

###Windows in azimuth, range, both, and a placeholder 
azi_window =  Matrix{Union{Missing, Float64}}(zeros(7,7))
range_window =  Matrix{Union{Missing, Float64}}(zeros(7,7))
standard_window = Matrix{Union{Missing, Float64}}(ones(7,7))
placeholder_window = Matrix{Union{Missing, Float64}}(ones(3,3))

azi_window[4,:] .= 1.
azi_window[5,:] .= 1.
azi_window[3,:] .= 1.

range_window[:, 4] .= 1.
range_window[:, 5] .= 1.
range_window[:, 3] .= 1.


aw = azi_window 
rw = range_window 
sw = standard_window
pw = placeholder_window


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
valid_funcs::Array{String} = ["AVG", "ISO", "STD"]
valid_derived_params::Array{String} = ["AHT", "PGG", "RNG", "NRG"]
FILL_VAL::Float64 = -32000.
RADAR_FILE_PREFIX::String = "cfrad"

##Threshold to exclude gates when gate >= value 
NCP_THRESHOLD::Float64 = .2 
PGG_THRESHOLD::Float64 = 1. 

REMOVE_LOW_NCP::Bool = true 
REMOVE_HIGH_PGG::Bool = true 

##Whether or not to replace a MISSING value with FILL in the spatial calculations 
REPLACE_MISSING_WITH_FILL::Bool = false 

placeholder_mask::Matrix{Bool} = [true true; false false]
