using ScikitLearn
using ScikitLearn.Pipelines
using JLD2
@sk_import decomposition: PCA
@sk_import linear_model: LinearRegression

pca = PCA()
lm = LinearRegression()

pip = Pipeline([("PCA", pca), ("LinearRegression", lm)])
#fit!(pip, ...)   # fit to some dataset
println("OK")
x = 3 
JLD2.save("pipeline.jld", "pip", pip)

# Load back the pipeline
pip = JLD2.load("pipeline.jld", "pip")
