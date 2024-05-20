# Basic Workflow Walkthrough 
RadarQC is a package that utilizes a methodology developed by [Dr. Alex DesRosiers and Dr. Michael Bell](https://journals.ametsoc.org/view/journals/aies/aop/AIES-D-23-0064.1/AIES-D-23-0064.1.xml) to remove non-meteorological gates from Doppler radar scans by leveraging machine learning techniques. In its current form, it contains functionality to derive a set of features from input radar data, use these features to train a Random Forest classification model, and apply this model to the raw fields contained within the radar scans. It also has some model evaluation ability. The beginning of this guide will walk through a basic workflow to train a model starting from scratch.  

  


---
## Preparing input data 
---

The first step in the process is to split our data so that some of it may be utilized for model training and the other portion for model testing. It's important to keep the two sets separate, otherwise the model may overfit. 

The basic requirement here is to have a directory or directories of cfradial scans, and two directories to put training and testing files, respectively. Make sure that no other files are present in these directories. To do this, the [`split_training_testing!`](https://github.com/irslushy/RadarQC.jl/blob/52d7f15ddb791bfef4341ff3e9d49fb1fb630049/src/RadarQC.jl#L576-L601) function will be used. For example, if one had two cases of radar data, located in `./CASE1/` and `./CASE2/` and wanted to split into `./TRAINING` and `./TESTING`, execute the command 
```julia
split_training_testing(["./CASE1", "./CASE2"], "./TRAINING", "./TESTING")
```
More information about this function is contained within the docs. 
</br></br>
Now that we have split the input files, we can proceed to calculate the features used to train and test our Random Forest model. Further details are contained within the aforementioned manuscript, but it has been shown that the parameters contained [here](https://github.com/irslushy/RadarQC.jl/blob/52d7f15ddb791bfef4341ff3e9d49fb1fb630049/MODELS/DesRosiers_Bell_23/config.txt) are most effective for discriminating between meteorological/non-meteorological gates. For this, we will use the [calculate_features](https://github.com/irslushy/RadarQC.jl/blob/52d7f15ddb791bfef4341ff3e9d49fb1fb630049/src/RadarQC.jl#L27-L95) function. Since we are calculating features to **train** a model at this point, we will assume that they have already a human apply QC. To get the most skillful model possible, we will want to remove "easy" cases from the training set, so set `REMOVE_LOW_NCP=true` and `REMOVE_HIGH_PGG=true` to ignore data not meeting minimum quality thresholds. It's also important to specify which variable contained with the input scans has already been QC'ed - in the ELDORA scans, this is `VG`. `missing` values must also be removed from the initial training set, so we'll use a raw variable `VV` to determine where these gates are located. With that said, one may now invoke 

```julia
calculate_features("./TRAINING", "./config.txt", "TRAINING_FEATURES.h5", true;
                    verbose=true, REMOVE_LOW_NCP = true, REMOVE_HIGH_PGG=true,
                    QC_variable="VG", remove_variable="VV")
```
To use the config file located at `./config.txt` to output the training features to `TRAINING_FEATURES.h5`. 

For clarity's sake, for the testing features this step would be exectued again but changing the input directory and output location. 

This is a somewhat computationally expensive process for large datasets >1000 scans, and so can take over 15 minutes. 
<br>

____

## Training a Random Forest Model 
___
<br>
Now that the somewhat arduous process of calculating input features has completed, it's time to train our model! We'll use the **training** set for this, which we have previously defined to be located at `./TRAINING_FEATURES.h5`. Invoke as follows

```julia
train_model("./TRAINING_FEATURES.h5", "TRAINED_MODEL.joblib"; verify=true, verify_out="TRAINING_SET_VERIFICATION.h5")
```

This will train a model based off the information contained within `TRAINING_FEATURES.h5`, saving it for further use at `./TRAINED_MODEL.joblib`. The `verify` keyword arguments means that, once trained, the model will be automatically applied to the training dataset and the predictions will be output, along with the ground truth/correct answers, to the h5 file at `TRAINING_SET_VERIFICATION.h5`. 

<br>

___
## Applying the trained model to a radar scan 
___
<br>

It's finally time to begin cleaning radar data! We'll use the `QC_scan` function to apply our model to raw moments. Let's imagine we want to clean the scan located at `./cfrad_example_scan` using the model we previously trained at `TRAINED_MODEL.joblib`. By default, this will clean the variables with the names `ZZ` and `VV`, though this can be changed by modifying the `VARIABLES_TO_QC` argument. They will be added to the cfradial file with the names `ZZ_QC` and `VV_QC`, respectively, though this suffix can be changed though keyword arguments. Execute as 
```julia
QC_scan("./cfrad_example_scan", "./config.txt", "./TRAINED_MODEL.joblib")
```



