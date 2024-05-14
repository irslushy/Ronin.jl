import math
import os, sys
import numpy as np
from scipy import ndimage
import netCDF4 as nc4
import re
import h5py

# Global constant so that we don't recompute that every time
# we call ...

EarthRadiusKm = 6375.636
EarthRadiusSquare = 6375.636 * 6375.636
DegToRad = 0.01745329251994372

# Beamwidth of ELDORA and TDR
eff_beamwidth = 1.8
beamwidth = eff_beamwidth*0.017453292;

# For doing average of neighbor values

avg_mask = np.ones((5,5))
avg_mask[2, 2] = 0

#add in for doing isolation calculation
iso_mask = np.ones((7,7))
iso_mask[3,3] = 0

#Changed over from older version to fix issue between km and m
def computeHtKmAirborn(elDeg, slantRange, aircraftHt):
    aircraftHtKm = aircraftHt/1000
    slantRangeKm = slantRange/1000
    elRad = elDeg * DegToRad
    term1 = slantRangeKm * slantRangeKm + EarthRadiusSquare
    term2 = 2 * slantRangeKm * EarthRadiusKm * math.sin(elRad)
    #return aircraftHtKm - EarthRadiusKm + math.sqrt(term1 + term2)
    ans = math.sqrt(term1 + term2) - EarthRadiusKm
    ans = ans + aircraftHtKm
    return ans

def probGroundGates(elDeg, slantRange, aircraftHt, azDeg, max_range):
    
    # Calculate the range where the beam would hit the ground
    # Define some values
    elev = elDeg * DegToRad
    az = azDeg * DegToRad
    sin_elev = math.sin(elev)
    tan_elev = math.tan(elev)
    radarAlt = aircraftHt
    range_m = slantRange 
    max_range_m = max_range
    earth_radius = EarthRadiusKm * 1000.0

    # Range that beam will intersect ground (from Testud et al. 1999)
    ground_intersect = (-(radarAlt)/sin_elev)*(1+radarAlt/(2*earth_radius*tan_elev*tan_elev))

    # Recast as relative to current range
    grange = ground_intersect - range_m
    
    # If the beam hits the ground too far away or the calculation fails then the beam cannot hit the ground
    #if ((ground_intersect >= max_range_m*2.5) or (ground_intersect <= 0)):
    #    gprob =  0.0
    
    # If the elevation angle is positive then the beam cannot hit the ground
    if (elDeg > 0):
        gprob =  0.0
    
    # If grange is 0 or negative then the beam is underground and assigned 1
    elif (grange <= 0):
        gprob =  1.0
    

    # If the range is less than the aircraft altitude then the beam cannot hit the ground
    elif (slantRange < aircraftHt):
        gprob =  0.0
        
    
    else:
        # Use simplified formula to solve for elevation angle where beam would
        # intersect ground at that range (gelev)
        # Note that this doesn't take into account earth curvature or topography
        gelev = math.asin(-radarAlt/range_m)
        eloffset = elev - gelev;
        
        # Fix issue that lets eloffset taper back to 0 underground
        if eloffset < 0:
            gprob = 1.0
        else:

            # Use an exponential function to model beam power
            beamaxis = math.sqrt(eloffset*eloffset);
            gprob = math.exp(-0.69314718055995*(beamaxis/beamwidth))

            if (gprob > 1.0):
                gprob = 1.0
    
    return gprob

def normalize(X):
    # Normalize values

    print('Normalizing value')
    min_maxs = np.empty((X.shape[0], 2))

    # print('X: ', X.shape)
    # print('min_maxs shape: ', min_maxs.shape)

    # Remember min and max we used so that we can normalize test data later

    for field in range(X.shape[0]):
        # print('X[', field, ']: ', X[field].shape)
        min = X[field].min()
        max = X[field].max()
        denominator = max - min
        # print('min_max[0]: ', min_maxs[0].shape)

        min_maxs[0] = X[field].min()
        min_maxs[1] = X[field].max()
        X[field] = (X[field] - min) / denominator

    return X, min_maxs


# loc is a file? load just this file
# loc is a dir? load all files in that dit

def expand_path(path, pattern = None):

    if os.path.isfile(path):
        return [ path ]

    file_list = []
    pattern = pattern or ".*"
    patt = re.compile(pattern)

    for root, dirs, files in os.walk(path, topdown = False):
        for name in files:
            if patt.match(name):
                file_list.append(os.path.join(root, name))
    return file_list

# Just as an image would be X * Y * 3 (3 values for RGB)
# our "images are X * Y * (size_of(my_vars) + 1)
#    +1 to add the altitude
#
#

# This is super slow as it has to open and close all the files.
# But it gives me the exact size I need to create an numpy array
# that doesn't have to be reallocated each time file data is appended
# to it...
# OK tradoff since we run this only once to convert to hdf5...

def get_dim(file_list):
    max_time = 0
    max_range = 0

    for path in file_list:
        nc_ds = nc4.Dataset(path, 'r')
        max_time = max(max_time, nc_ds.dimensions['time'].size)
        max_range = max(max_range, nc_ds.dimensions['range'].size)
        nc_ds.close()
    return max_time, max_range

#
# Split X and Y into X_train, Y_train, X_test, Y_test
# This is no longer used, but being retained for now

def split_dataset(X, Y, split = 90):
    count = Y.shape[0]
    cutoff = int(Y.shape[0] * split / 100)

    X_train, X_test = X[:, :cutoff], X[:, cutoff:]
    Y_train, Y_test = Y[:cutoff], Y[cutoff:]

    return X_train, X_test, Y_train, Y_test


# Fill expected result by comparing VT to VG
# O: original field
# E: Edited field
# return: numpy array of booleans: True if the value is not the fill_val

# If VG is equal to fill val -> bad data was removed
# else -> good data
#This function does not need O passed but that change will be made later

#This function determines if a radar gate was retained during manual QC
def fill_expected(O, E, fill_val):
    # return ((O == E) or (E != fill_val)).astype(int)
    # return (O == E).astype(int)
    return (E != int(-32768)).astype(int)


#This is the original conditional before fill_val corrections were considered
#def fill_expected(O, E, fill_val):
#    return (O == E).astype(int)

# Compute neighbor average for a given variable
# var: 2D variable from the netcdf file
# return: numpy array of averages

def compute_avgs(var, fill_val):
    return ndimage.generic_filter(var, np.nanmean, footprint = avg_mask,
                                  mode = 'constant', cval = np.NaN)

def compute_std(var, fill_val):
    return ndimage.generic_filter(var, np.nanstd, footprint = avg_mask,
                                  mode = 'constant', cval = np.NaN)

#This tells the model about the isolation of radar gates

def NanParamCalc(var):
    fill_val = int(-32768)
    index = (var != fill_val).astype(int)
    return np.nanmean(index)

def compute_isolation(var, fill_val):
    return ndimage.generic_filter(var, NanParamCalc, footprint = iso_mask, mode = 'constant', cval = np.NaN)

# The notes here apply to a previous configuration
# Simple for now
# X: (9, *) array  (ZZ, VV, SW, NCP, ALT, AZZ, AVV, ASW, ANCP)
# Y: (1, *) array
#
# TODO pass fields as arguments...


#If a predictor is added it must be added here to this list initially
#This has been done in the commented out section, just remove the bracket
#from previous line and add a comma after iso
model_fields = ['VEL', 'DBZ', 'SQI',
                'ALT',
                'AVV', 'AZZ', 'ASQI',
                'SVV', 'SZZ', 'SSQI',
                'ISO',
                'PGG',
		        'RG',
		        'NRG']

#Edit this field when you have selected a more narrow list of predictors from Lasso
model_fields_trim = ['DBZ','SQI','ALT','AVV','PGG','NRG']

def field_names():
    return model_fields

def field_names_trim():
    return model_fields_trim

def remove_field(name):
    if name in model_fields:
        model_fields.remove(name)
    else:
        print('No such field ', name)

def load_netcdf(loc, pattern = None):
    # The first variable is the original (pre-clean) key
    # The "result' variabe is the cleaned up field.
    # To create Y, we compare the "key" to the "cleaned" variable.

    # ALT and all the averages are added by the script.
    # Note that once we compute Y, we remove the cleaned up field
    #     because it isn't needed anymore for training or testing.


    my_vars = [ 'VEL', 'DBZ', 'SQI', 'VE']
    my_avgs = [ 'AVV', 'AZZ', 'ASQI' ]
    my_stds = [ 'SVV', 'SZZ', 'SSQI' ]
    my_iso = ['ISO']
    my_PGG = ['PGG']
    my_RG = ['RG']
    my_NRG = ['NRG']
    result_var = 'VE'
    alt_index = len(my_vars)        # Index of altitude
    avg_offset = alt_index + 1      # Averages will be after the altitude
    std_offset = avg_offset + len(my_avgs)
    iso_index = std_offset + len(my_stds)
#Uncomment the below line to add PGG
    PGG_index = iso_index + 1
    RG_index = PGG_index + 1
    NRG_index = RG_index + 1


    file_list = expand_path(loc, pattern)
    num_files = len(file_list)

    max_x, max_y = get_dim(file_list)
    flat_size = max_x * max_y
    num_cols = num_files * flat_size
#add a 1 to dim for the PGG once it is added
    dim = (len(my_vars) + 1 + len(my_avgs) + len(my_stds) + 1 + 1 + 1 + 1, num_cols)

    X = np.empty(dim)
    # DBZ = np.empty(num_cols)
    ob_index = 0
    fill_val = None

    for path in file_list:
        print ('Reading ', path)
        nc_ds = nc4.Dataset(path, 'r')

        # read the variables we need to compute altitude
        max_time = nc_ds.dimensions['time'].size
        max_range = nc_ds.dimensions['range'].size
        rnge = nc_ds.variables['range']
        plane_alts = nc_ds.variables['altitude'] # (time)
        ray_angles = nc_ds.variables['elevation'] # (time)
        ranges = nc_ds.variables['range'] # (time)
        azimuths = nc_ds.variables['azimuth'] # (time)

        # Read the "feature" variables

        for var in range(len(my_vars)):
            my_var = nc_ds.variables[my_vars[var]]
            fill_val = getattr(my_var, '_FillValue')

            # Flatten it and append to the X array
            one_d = (my_var[:]).reshape(-1)
            X[var, range(ob_index, ob_index + one_d.size)] = one_d

            # Compute the neighbor average, and append it to X

            if var >= 3:        # Don't average VG since we will remove it
                continue

            print('Computing averages for ', my_vars[var])
            var_avrg = compute_avgs(my_var, fill_val)
            one_d = (var_avrg[:]).reshape(-1)
            X[var + avg_offset, range(ob_index, ob_index + one_d.size)] = one_d

            print('Computing std deviations for ', my_vars[var])
            var_std = compute_std(my_var, fill_val)
            one_d = (var_std[:]).reshape(-1)
            X[var + std_offset, range(ob_index, ob_index + one_d.size)] = one_d

        #This section calculates the isolation of radar gates
        print('Computing isolation of ', my_vars[1])
        var_iso = compute_isolation(nc_ds.variables[my_vars[1]], fill_val)
        one_d_iso = (var_iso[:]).reshape(-1)
        X[iso_index, range(ob_index, ob_index + one_d.size)] = one_d_iso

        # Compute and add the height

        print('Computing altitudes and probability of ground gates for ', max_time * max_range, ' entries')
#        heights = np.ones( (max_time, max_range) )  # Comment this if the following section is uncommented, this line creates a null altitude field to save time during testing

        #This is where altitude is computed and how probability of ground gates should be implemented
        heights = np.empty( (max_time, max_range) )
        probgg = np.empty( (max_time, max_range) )

        for time_idx in range(max_time):
             for range_idx in range(max_range):
                 tangle = float(ray_angles[time_idx].data)
                 trange = float(ranges[range_idx].data)
                 talt   = float(plane_alts[time_idx].data)
                 taz    = float(azimuths[time_idx].data)

                 heights[time_idx, range_idx] = computeHtKmAirborn(
                    tangle, trange, talt)

                 probgg[time_idx, range_idx] = probGroundGates(
                    tangle, trange, talt, taz, max_range)

        one_d = (heights[:]).reshape(-1)
        X[alt_index, range(ob_index, ob_index + one_d.size)] = one_d

        one_d = (probgg[:]).reshape(-1)
        X[PGG_index, range(ob_index, ob_index + one_d.size)] = one_d

        print('Computing Range and Normalized Range')
        rg = np.empty( (max_time, max_range) )
        nrg = np.empty( (max_time, max_range) )
        for time_idx in range(max_time):
             for range_idx in range(max_range):
                 rg[time_idx, range_idx] = rnge[range_idx]
                 ralt = float(plane_alts[time_idx].data)
                 nrg[time_idx, range_idx] = (rnge[range_idx])/ralt

        one_d = (rg[:]).reshape(-1)
        X[RG_index, range(ob_index, ob_index + one_d.size)] = one_d

        one_d = (nrg[:]).reshape(-1)
        X[NRG_index, range(ob_index, ob_index + one_d.size)] = one_d

        nc_ds.close()
        #The below line is vital to continued operation with multiple files
        ob_index += one_d.size


    # Remove columns where VT is fill_val (Should that be done earlier?)
    #  as we are processing each file?
    print(np.shape(X))
    X = X[:, X[0] != fill_val]

    # Randomize - this step was removed as it was unnecesary and problematic
    #print ('Randomizing the data')
    #X = np.random.permutation(X.T).T

    # print("X after: ", X.shape)
    #Create the classification array, it checks if a gate was retained or not
    Y = fill_expected(X[my_vars.index('VEL'),:],
                      X[my_vars.index('VE'),:],
                      fill_val)

    # Delete the VG row from X (since it is how Y is calculated)
    X = np.delete(X, my_vars.index('VE'), 0)

    # Maybe delete more fields we don't want to deal with
    # Example: AZZ
    #     X = np.delete(X, my_vars.index('ZZ'), 0)
    # Also need to remove it from field names.
    #     delete_fields('ZZ')
    print(np.shape(X))
    print(np.shape(Y))
    return X, Y

#
# Load X and Y from the given h5 file
#

def load_h5(loc):
    dataset = h5py.File(loc, 'r')
    X = np.array(dataset['X'][:])
    Y = np.array(dataset['Y'][:])

    print('X: ', X.shape)
    print('Y: ', Y.shape)

    return X, Y

#
# Precision, recall, and f1_score
# Not implemented yet
#

def precision(matrix):
    return 1

def recall(matrix):
    return 1

def f1_score(prec, recl):
    return 2 * prec * recl / (prec + recl)
