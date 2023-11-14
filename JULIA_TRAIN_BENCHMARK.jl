using NCDatasets
using HDF5
using DataFrames
using TypedTables
using DecisionTree
using BenchmarkTools

function train(x, y)
    ###Hyperparameters from paper: 21 trees, max depth 14
    currmodel = DecisionTree.RandomForestClassifier(;n_trees=21, max_depth=14)
    ###Determine the class weights 
    fit!(currmodel, x[:,:], y[:])
end

filepath = "./data/raw_Vortex.h5"
currset = HDF5.h5open(filepath)
x = currset["X"]
y = currset["Y"]

bench = @benchmark train(x,y) 
io = IOBuffer()
show(io, "text/plain", bench)

print(stdout, String(take!(io)))
