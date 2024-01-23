# JMLQC

JMLQC, or Julia Machine Learning Quality Control, is a combination julia/python implementation of the algorithm described in [DesRosiers and Bell 2023](https://journals.ametsoc.org/view/journals/aies/aop/AIES-D-23-0064.1/AIES-D-23-0064.1.xml) for removing non-meteoroloigcal gates from airborne radar scans. Care has been taken to ensure relative similarity to the form described in the manuscript, but some changes have been made in the interest of computational speed. 

A key part of the process is computing necessary derived parameters from the raw radar moments, which may be custom-specified in a parameters file. Many of the relevant functions for these calculations are contained within [JMLQC_utils.jl](./src/JMLQC_utils.jl). For preprocessing a single radar scan that has already been manually-QC'ed, likely for use in an already trained random forest model, or an entire directory of manually-QC'ed scans, likely for use in training a new model, and outputting the resultant features to an h5 file, see [calculate_cfrad_parameters.jl](./calculate_cfrad_parameters.jl).

For applying the algorithm using an existing/trained RF model, see [QC_single_scan.jl](./QC_single_scan.jl) 
  
  <br> 
  <br> 

___
## Guide: Processing new data, training, and evaluating a new model
___
  <br>
  
The first step in training a new random forest model is determining which portions of the data will be used for training, testing, and validation. A helpful script here is [split_data.jl](./split_data.jl) - this can be used to automatically split a collection of scans into a training directory and a testing directory. In order for the script to be configured properly, the variables relating to the different paths at the top of the file must be modified by the user. 
<br> <br>The current configuration is consistent with the 80/20 training/testing split described in the manuscript, as well as to have an equal number of scans from each "case" represented in the testing set. It is expected that the script would work for different training/testing splits, but this has not yet been tested. <br><br>

Once the training and testing scans have been placed into separate directories, data processing may begin. [calculate_RF_Input_Features.jl](calculate_RF_Input_Features.jl) will be the primary script utilized here. The script processes a directory (or single scan) of scans, and outputs the calculated features into an .h5 file, with the desired features specified by the user in a text file. <br><br>

For the case where training scans are located within `/cfradials/training/`, the desired features to be calculated are specified in `features.txt`, and you wish to output the input features to `training_set.h5`, the script would be run as
```
juila calculate_RF_Input_Features.jl /cfradials/training -m D -f features.txt -o training_set.h5
```
If you wish to remove a validation set from the training dataset, utilize [remove_validation_from_training.jl](./remove_validation_from_training.jl)
<br><br>


Finally, we can train a model to process our data. To do so, utilize [train_RF_model_no_split.jl](train_RF_model_no_split.jl). If training data is contained within `training_set.h5`, and you wish to name your trained model `trained_model.bson`, invoke as follows. 
```
julia train_RF_model_no_split.jl training_set.h5 trained_model.bson
```
<b>NOTE: This may take on the order of 20-30 minutes</b><br><br>
This script also includes the option to verify the model on the training set and output the results to a separate h5 file. If you wish to do this, execute the same as above, but include `-v true` and `-o path_to_desired_output_location.h5`<br><br><br>
___
## Guide: Applying algorithm to a set of scans using a trained model 
___
To remove non-meteorological echoes from scan or set of scans, utilize [QC_scan.jl](QC_scan.jl). To properly configure, the `VARIALBES_TO_QC` array must be modified to contain the names of the variables within each scan you wish to apply the random forest to. 
___
## Notes on data conventions
_______
For the verification 'Y' array in the training scripts I have adopted the convention that 1 indicates METEOROLOGICAL DATA, and 0 indicates NON_METEOROLOGICAL DATA 

Furthermore, for QC-ed variables in the output files, the following is adopted:  
    UNDEF/MISSING: MISSING DATA IN ORIGINAL FILE  
    0: REMOVED IN MLQC (AFTER BASE THRESHOLDS WERE APPLIED)  
    VALUE: RETAINED IN MLQC  

Data is written out to NetCDF files to be CF-Compliant in Julia and other column-major languages, such that it has dimensions of (range x time). 
However, if it were loaded in a row-major language, such as python, it would take on dimensions of (time x range). 


