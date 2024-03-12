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

export_path="../../data/raw"

python ../../../../IsaacGymEnvs/isaacgymenvs/train.py task=A1Terrain_NeuroRL_exp \
  train=A1Terrain_PPO_LSTM_NeuroRL \
  capture_video=False \
  capture_video_len=1000 \
  force_render=True \
  headless=False \
  test=True \
  checkpoint=../../models/${model_name}/nn/model.pth \
  num_envs=400 \
  task.env.specifiedCommandVelocityRanges.linear_x='[1.0, 1.0]' \
  task.env.specifiedCommandVelocityRanges.linear_y='[0., 0.]' \
  task.env.specifiedCommandVelocityRanges.yaw_rate='[0., 0.]' \
  task.env.specifiedCommandVelocityN.linear_x=1 \
  task.env.specifiedCommandVelocityN.linear_y=1 \
  task.env.specifiedCommandVelocityN.yaw_rate=1 \
  task.env.specifiedCommandVelocityN.n_copies=400 \
  task.env.export_data=false \
  task.env.export_data_actor=false \
  task.env.export_data_critic=false \
  task.env.evaluate.perturbPrescribed.perturbPrescribedOn=True \
  task.env.evaluate.perturbPrescribed.forceY=-3. \
  task.env.export_data_path=${export_path} \
  +output_path=${export_path}/${script_name}