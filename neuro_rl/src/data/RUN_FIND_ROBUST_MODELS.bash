#!/bin/bash

# Define a function that takes a file path as an argument
process_file() {
  local file_path="$1"
  local file_name=$(basename "$file_path")    
  echo "$file_path/$file_name"
}

# Define an array of model names
models_info=(
  # Could go and re-select most robust models in each folder...?
  "A1Terrain_PPO_LSTM_NeuroRL:A1Terrain_NeuroRL_exp:A1-1.0MASS-LSTM16-TERR-01" # need to rerun training for longer epochs (model did not yet converge)

  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-0.5MASS-LSTM16-TERR-01"

  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-BASELINE-01"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-DIST-01"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-TERR-01"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-DISTTERR-01"
  
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-BASELINE-01"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-DIST-01"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-TERR-01" # see if there is any difference between these two in terms of FPs, config is identical
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-TERR-201" # see if there is any difference between these two in terms of FPs, config is identical
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-DISTTERR-01"
)

export_path="../../data/raw"

# Loop through the model names and call the sub-script for each one
for model_info in "${models_info[@]}"
do
  echo "Running model: $model_info"

  # Split model_info into its components
  IFS=':' read -r train_file task_file model_name <<< "$model_info"

  echo "MODEL PROCESSING:"
  echo "Train File: $train_file"
  echo "Task File: $task_file"
  echo "Model Name: $model_name"

  search_dir="../../models/$model_name/nn" 

  # Use 'find' to locate all files, sort them alphanumerically, and then read them line by line
  find "$search_dir" -type f | sort -V | while read file; do
      process_file "$file"
  done
done
# echo "$file_name"