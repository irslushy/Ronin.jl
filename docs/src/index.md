# API / Function References 

## Calculating Model Input Features  

```@docs 
calculate_features(::String, ::String, ::String, ::Bool)
calculate_features(::String, ::Vector{String}, ::Vector{Matrix{Union{Missing, Float64}}}, ::String, ::Bool)
split_training_testing!
train_model(::String, ::String)
``` 

## Applying and evaluating a trained model to data 

```@docs
QC_scan
predict_with_model(::String, ::String)
evaluate_model(::String, ::String, ::String)
```

## Non-user facing

```@docs
get_task_params
process_single_file
```
