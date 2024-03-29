# API / Function References 

## Calculating Model Input Features  

```@docs 
calculate_features(::String, ::Vector{String}, ::Vector{Matrix{Union{Missing, Float64}}}, ::String, ::Bool)
calculate_features(::String, ::String, ::String, ::Bool)
split_training_testing!
``` 

## Applying a trained model to data 

```@docs
QC_scan
```

## Non-user facing

```@docs
get_task_params
process_single_file
```
