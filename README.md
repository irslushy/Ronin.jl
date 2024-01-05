# JMLQC

JMLQC, or Julia Machine Learning Quality Control, is a combination julia/python implementation of the algorithm described in [DesRosiers and Bell 2023](https://journals.ametsoc.org/view/journals/aies/aop/AIES-D-23-0064.1/AIES-D-23-0064.1.xml) for removing non-meteoroloigcal gates from airborne radar scans. Care has been taken to ensure relative similarity to the form described in the manuscript, but some changes have been made in the interest of computational speed. 

A key part of the process is computing necessary derived parameters from the raw radar moments, which may be custom-specified in a parameters file. Many of the relevant functions for these calculations are contained within [utils.jl](./utils.jl). For preprocessing a single radar scan that has already been manually-QC'ed, likely for use in an already trained random forest model, or an entire directory of manually-QC'ed scans, likely for use in training a new model, and outputting the resultant features to an h5 file, see [calculate_cfrad_parameters.jl](./calculate_cfrad_parameters.jl).

For applying the algorithm using an existing/trained RF model, see [QC_single_scan.jl](./QC_single_scan.jl) 


CONVENTIONS: For the "verification 'Y'" array in the training scripts, I have adopted the convention that 1 indicates METEOROLOGICAL DATA, and 0 indicates NON_METEOROLOGICAL DATA 
