using RadarQC
using BenchmarkTools

printstyled("OK\n", color=:red)
###Hyperparameters from paper: 21 trees, max depth 14
# currmodel = DecisionTree.RandomForestClassifier(;n_trees=21, max_depth=14)
# ###Determine the class weights 

# filepath = "./data/raw_Vortex.h5"
# currset = HDF5.h5open(filepath)
# x = currset["X"]
# y = currset["Y"]

# fit!(currmodel, x[:,:], y[:])

