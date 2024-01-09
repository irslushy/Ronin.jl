# JMLQC

JMLQC, or Julia Machine Learning Quality Control, is a combination julia/python implementation of the algorithm described in [DesRosiers and Bell 2023](https://journals.ametsoc.org/view/journals/aies/aop/AIES-D-23-0064.1/AIES-D-23-0064.1.xml) for removing non-meteoroloigcal gates from airborne radar scans. Care has been taken to ensure relative similarity to the form described in the manuscript, but some changes have been made in the interest of computational speed. 

A key part of the process is computing necessary derived parameters from the raw radar moments, which may be custom-specified in a parameters file. Many of the relevant functions for these calculations are contained within [utils.jl](./utils.jl). For preprocessing a single radar scan that has already been manually-QC'ed, likely for use in an already trained random forest model, or an entire directory of manually-QC'ed scans, likely for use in training a new model, and outputting the resultant features to an h5 file, see [calculate_cfrad_parameters.jl](./calculate_cfrad_parameters.jl).

For applying the algorithm using an existing/trained RF model, see [QC_single_scan.jl](./QC_single_scan.jl) 


CONVENTIONS: For the "verification 'Y'" array in the training scripts, I have adopted the convention that 1 indicates METEOROLOGICAL DATA, and 0 indicates NON_METEOROLOGICAL DATA 

Furthermore, for QC-ed variables in the output files, the following is adopted:  
    UNDEF/MISSING: MISSING DATA IN ORIGINAL FILE  
    0: REMOVED IN MLQC (AFTER BASE THRESHOLDS WERE APPLIED)  
    VALUE: RETAINED IN MLQC  

Data is written out to NetCDF files to be CF-Compliant in Julia and other column-major languages, such that it has dimensions of (range x time). 
However, if it were loaded in a row-major language, such as python, it would take on dimensions of (time x range). 



Steps for creating a training dataset, training a random forest model, and evaluating its performance:  

For creating a training dataset, it is assumed that the input data has already been split into training and testing compoments. Place the set of training files and the set of testing files into their own respective directories.  

Then, utilize the [calculate_cfrad_parameters.jl](./calculate_cfrad_parameters.jl) script in directory processing mode to output data to a training.h5 set. For example, if training data is located in `./training_data/`, the features to be calculated for use in the model are specified in `./config.txt`, and you wish to output the data to `./training_data.h5`, execute
```
julia ./calculate_cfrad_parameters ./training_data/ -f ./config.txt -o ./training_data.h5 -m D
```

To train a model on this data, now utilize [train_RF_model_no_split.jl](./train_RF_model_no_split.jl)
