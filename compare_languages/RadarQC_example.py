###Python script adapted from Dr. Alex DesRosiers' P3 MLQC codebase 
###Designed to somewhat emulate the julia version of the functionality in order to 
###Facilitate apples-to-apples comparisons of timing for the quality control process 

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

def load_single_netcdf_p3_new(loc, pattern = None, vars_to_calc=['VEL', 'DBZ', 'AVEL', 'ADBZ', 'SVEL', 'ISO', 'PGG', 'RG', 'NRG']):
    # The first variable is the original (pre-clean) key
    # The "result' variabe is the cleaned up field.
    # To create Y, we compare the "key" to the "cleaned" variable.

    # ALT and all the AVGs/SDs are added by the script.
    # Note that once we compute Y, we remove the cleaned up field
    #     because it isn't needed anymore for training or testing.


    my_vars = [ 'VEL', 'DBZ', 'SQI']
    my_avgs = [ 'AVEL', 'ADBZ', 'ASQI' ]
    my_stds = [ 'SVEL', 'SDBZ', 'SSQI' ]
    my_iso = ['ISO']
    my_PGG = ['PGG']
    my_RG = ['RG']
    my_NRG = ['NRG']


    ISO_VAR = 'DBZ' 

    alt_index = len(my_vars)        # Index of altitude
    avg_offset = alt_index + 1      # Averages will be after the altitude
    std_offset = avg_offset + len(my_avgs)
    iso_index = std_offset + len(my_stds)
    PGG_index = iso_index + 1
    RG_index = PGG_index + 1
    NRG_index = RG_index + 1



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

        ALT_COMPLETE = False 

        PGG_matrix = [[]]
        ALT_matrix = [[]]

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
                    

            elif var == 'RG': 
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

num_reps = 5
for path in CASES:
    curr_files = os.listdir(path)

    for file in curr_files: 
        ##Try each scan 10 times 
        curr_timer = timeit.Timer(lambda: load_single_netcdf_p3_new(path + "/" + file, vars_to_calc=['ISO', 'AVEL', 'DBZ']))
        curr_duration = curr_timer.timeit(num_reps)
        print("AVERAGE TIME FOR THIS SCAN " + str(curr_duration / num_reps) + " s")