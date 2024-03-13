#!/bin/bash

# Define an array of model names
model_names=(
    # "ANYMAL-1.0MASS-LSTM16-DISTTERR-00"
    # "ANYMAL-1.0MASS-LSTM16-DISTTERR-01"
    # "ANYMAL-1.0MASS-LSTM16-DISTTERR-02"
    # "ANYMAL-1.0MASS-LSTM16-TERR-00"
    # "ANYMAL-1.0MASS-LSTM16-TERR-01"
    # "ANYMAL-1.0MASS-LSTM16-TERR-02"

    # "ANYMAL-1.0MASS-LSTM4-TERR-01"
    # "ANYMAL-1.0MASS-LSTM16-TERR-01"
    # "2024-03-09-13-06_AnymalTerrain"
    # "ANYMAL-1.0MASS-LSTM16-DISTTERR-01"
    # "ANYMAL-1.0MASS-LSTM16-DISTTERR-01"
    # "ANYMAL-1.0MASS-LSTM4-DISTTERR-01"
    # "ANYMAL-1.0MASS-LSTM4-TERR-01"
    # "ANYMAL-1.0MASS-LSTM16-TERR-201"
    # "A1-1.0MASS-LSTM16-DISTTERR-203"
    # "A1-1.0MASS-LSTM16-TERR-207"
    # "A1TERRAIN-1.0MASS-LSTM16-TERR-218"
    # "A1-1.0MASS-LSTM16-TERR-228"
    # "A1-1.0MASS-LSTM16-TERR-01"
    "ANYMAL-0.5MASS-LSTM16-TERR-01"
    
)

# Loop through the model names and call the sub-script for each one
for model_name in "${model_names[@]}"
do
    echo "Running model: $model_name"
    ./RUN_UNDISTURBED_ANYMAL.bash "$model_name"
done