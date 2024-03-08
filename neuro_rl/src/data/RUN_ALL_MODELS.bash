#!/bin/bash

# Define an array of model names
model_names=(
    "ANYMAL-1.0MASS-LSTM16-DISTTERR-00"
    "ANYMAL-1.0MASS-LSTM16-DISTTERR-01"
    "ANYMAL-1.0MASS-LSTM16-DISTTERR-02"
    "ANYMAL-1.0MASS-LSTM16-TERR-00"
    "ANYMAL-1.0MASS-LSTM16-TERR-01"
    "ANYMAL-1.0MASS-LSTM16-TERR-02"
)

# Loop through the model names and call the sub-script for each one
for model_name in "${model_names[@]}"
do
    echo "Running model: $model_name"
    ./RUN_UNPETURBED.bash "$model_name"
done