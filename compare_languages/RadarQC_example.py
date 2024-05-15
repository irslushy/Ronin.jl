###Python script adapted from Dr. Alex DesRosiers' P3 MLQC codebase 
###Designed to somewhat emulate the julia version of the functionality in order to 
###Facilitate apples-to-apples comparisons of timing for the quality control process 

###Currently only features functionality to compute features as input to model, 
###as this is the most time-consuming part of the workflow and so should be informative 

###NOTE: Currently only configured to work with the NOAA P3 TDR data - will work on 
###Expanding this to the ELDORA dataset 

import math
import os, sys
import numpy as np
#Workarounds for new numpy install - Uncomment Below if needed
#np.object = object
#np.int = int
#np.float = float
#np.bool = bool
#np.typeDict = np.sctypeDict
from scipy import ndimage
import netCDF4 as nc4
import re
import h5py
import glob

import sys, argparse
import pandas as pd
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

import matplotlib as mpl
import matplotlib.pyplot as plt

from utils import *

##Isaac module additions
from os import listdir 
import timeit 


####Inputs: 
####loc: path to cfradial file or directory of cfradials to process 
####pattern:  secondary input to expand_path (unimportant)
####vars_to_calc: List of features to calculate for each cfradial file.
####Acceptable values: VEL, DBZ, SQI, AVEL, ADBZ, ASQI, SVEL, SDBZ, SSQI, PGG, RNG, NRG, AHT
####Where VEL, DBZ, SQI are raw fields, the prefix A(VEL) indicates a spatial average, 
####The prefix S(VEL) indicates a spatial standard deviation, PGG is probabilty of ground gate, 
####RNG is range, NRG is range normalized by the height of the aircraft
def load_single_netcdf_p3_new(loc, pattern = None, vars_to_calc=['VEL', 'DBZ', 'AVEL', 'ADBZ', 'SVEL', 'ISO', 'PGG', 'RG', 'NRG']):
   
    my_vars = [ 'VEL', 'DBZ', 'SQI']
    my_avgs = [ 'AVEL', 'ADBZ', 'ASQI' ]
    my_stds = [ 'SVEL', 'SDBZ', 'SSQI' ]

    ISO_VAR = 'DBZ' 

    #ELEV_index = NRG_index + 1


    file_list = expand_path(loc, pattern)
    print(file_list)
    num_files = len(file_list)

    max_x, max_y = get_dim(file_list)
    flat_size = max_x * max_y
    num_cols = num_files * flat_size
    dim = (len(vars_to_calc), num_cols)

    X = np.empty(dim)
    ob_index = 0
    fill_val = None

    ###ISAAC MODIFICATIONS 

    for path in file_list:

        nc_ds = nc4.Dataset(path, 'r')

        # read the variables we need to compute altitude
        max_time = nc_ds.dimensions['time'].size
        max_range = nc_ds.dimensions['range'].size
        rnge = nc_ds.variables['range']
        plane_alts = nc_ds.variables['altitude'] # (time)
        ray_angles = nc_ds.variables['elevation'] # (time)
        ranges = nc_ds.variables['range'] # (time)
        azimuths = nc_ds.variables['azimuth'] # (time)


        ###Isaac modifications 
        ###This is clunky as heck but just wanted to get something working 
        for i, var in enumerate(vars_to_calc): 

            if var in my_vars: 

                currmoment = nc_ds.variables[var]
                fill_val = getattr(currmoment, '_FillValue')

                # Flatten it and append to the X array
                one_d = (currmoment[:]).reshape(-1)
                X[i, range(ob_index, ob_index + one_d.size)] = one_d

            elif var in my_avgs:

                #print('Computing averages for ', var)
                var_avrg = compute_avgs(nc_ds.variables[var[1:]], fill_val)
                one_d = (var_avrg[:]).reshape(-1)
                X[i, range(ob_index, ob_index + one_d.size)] = one_d

            elif var in my_stds:

                #print('Computing stds for ', var)
                var_std = compute_std(nc_ds.variables[var[1:]], fill_val)
                one_d = (var_std[:]).reshape(-1)
                X[i, range(ob_index, ob_index + one_d.size)] = one_d 
            
            elif var == 'ISO': 

                #This section calculates the isolation of radar gates
                #print('Computing isolation of ', var)
                var_iso = compute_isolation(nc_ds.variables[ISO_VAR], fill_val)
                one_d = (var_iso[:]).reshape(-1)
                X[i, range(ob_index, ob_index + one_d.size)] = one_d

            elif var == 'AHT': 
                heights = np.empty( (max_time, max_range) )
              
                for time_idx in range(max_time):
                    for range_idx in range(max_range):
                        tangle = float(ray_angles[time_idx].data)
                        trange = float(ranges[range_idx].data)
                        talt   = float(plane_alts[time_idx].data)
                        taz    = float(azimuths[time_idx].data)

                        heights[time_idx, range_idx] = computeHtKmAirborn(
                            tangle, trange, talt)
                
                one_d = (heights[:]).reshape(-1)
                X[i, range(ob_index, ob_index + one_d.size)] = one_d 

            elif var == 'PGG':
                probgg = np.empty( (max_time, max_range) )
            
                for time_idx in range(max_time):
                    for range_idx in range(max_range):
                        tangle = float(ray_angles[time_idx].data)
                        trange = float(ranges[range_idx].data)
                        talt   = float(plane_alts[time_idx].data)
                        taz    = float(azimuths[time_idx].data)

                        probgg[time_idx, range_idx] = probGroundGates(
                            tangle, trange, talt, taz, max_range)

                
                one_d = (probgg[:]).reshape(-1)
                X[i, range(ob_index, ob_index + one_d.size)] = one_d
                    

            elif var == 'RNG': 
                rg = np.empty( (max_time, max_range) )

                for time_idx in range(max_time):
                    for range_idx in range(max_range):
                        rg[time_idx, range_idx] = rnge[range_idx]
                        #elev[time_idx, range_idx] = rnge[time_idx]
                        ralt = float(plane_alts[time_idx].data)

                one_d = (rg[:]).reshape(-1)
                X[i, range(ob_index, ob_index + one_d.size)] = one_d

            elif var == 'NRG': 

                nrg = np.empty( (max_time, max_range) )
                #elevat = np.empty( (max_time, max_range) )
                for time_idx in range(max_time):
                    for range_idx in range(max_range):
                        rg[time_idx, range_idx] = rnge[range_idx]
                        #elev[time_idx, range_idx] = rnge[time_idx]
                        ralt = float(plane_alts[time_idx].data)
                        nrg[time_idx, range_idx] = (rnge[range_idx])/ralt

                one_d = (nrg[:]).reshape(-1)
                X[i, range(ob_index, ob_index + one_d.size)] = one_d
                # Read the "feature" variables


            else: 
                raise Exception("UNKNOWN VARIABLE NAME " + str(var)) 
            

        nc_ds.close()
        #The line below is vital to continued operation with multiple files
        ob_index += one_d.size
    
    print(np.shape(X))
    return X



###CHANGE THESE PATHS IN ORDER TO POINT TO CORRECT TRAINING DATASETS 
###Currently have them set to point toward the benchmarking cfradials included in the github repository 
CASE_DIR = "../BENCHMARKING/NOAA_benchmark_cfrads/"
CASES = [CASE_DIR]

###Number of times to benchmark each file/directory 
num_reps = 5
###Features to calculate 
vars_to_calc = ['SQI', 'AHT', 'SVEL', 'PGG', 'RNG', 'ISO']

for path in CASES:
    curr_files = os.listdir(path)

    for file in curr_files: 
        ##Try each scan 10 times 
        curr_timer = timeit.Timer(lambda: load_single_netcdf_p3_new(path + "/" + file, vars_to_calc=vars_to_calc))
        curr_duration = curr_timer.timeit(num_reps)
        print("\nAVERAGE TIME FOR THIS SCAN " + str(curr_duration / num_reps) + " s\n")