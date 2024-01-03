# JMLQC

JMLQC, or Julia Machine Learning Quality Control, is a combination julia/python implementation of the algorithm described in [DesRosiers and Bell 2023](https://journals.ametsoc.org/view/journals/aies/aop/AIES-D-23-0064.1/AIES-D-23-0064.1.xml) for removing non-meteoroloigcal gates from airborne radar scans. 

The process begins by computing necessary derived parameters from the raw radar moments, which may be custom-specified in a parameters file. Many of the relevant functions for these calculations are contained within [utils.jl](./utils.jl)
