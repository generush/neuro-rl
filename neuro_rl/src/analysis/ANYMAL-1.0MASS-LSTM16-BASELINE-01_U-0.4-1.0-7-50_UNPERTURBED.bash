#!/bin/bash

# Extract the base name of the script
script_name=$(basename "$0")

# GET THE MODEL NAME
#----------------------------------------------------------------
# Save the old IFS value
oldIFS="$IFS"

# Set IFS to underscore for splitting the filename
IFS='_'

# Read the split parts into an array
read -ra ADDR <<< "$script_name"

# Restore the old IFS
IFS="$oldIFS"

# Create a string that includes the first part of the filename
model_name=${ADDR[0]}

#----------------------------------------------------------------

# Optionally remove the .sh extension from the script name
# If your script file doesn't have an extension or you want to keep it, you can skip this step
script_name="${script_name%.*}"

# export_path="../../data/raw"

python ../../analysis_pipeline.py --config_path "../../cfg/analyze/analysis.yaml" --input_path "../data/${script_name}" --output_path "../data/${script_name}"