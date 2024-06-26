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

python ../../../../IsaacGymEnvs/isaacgymenvs/train.py task=bash_AnymalTerrain_NeuroRL_exp \
  train=AnymalTerrain_PPO_LSTM_NeuroRL \
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
  task.env.ablate.wait_until_disturbance=false \
  task.env.ablate.random_trial=false \
  task.env.ablate.random.obs_in= 0 \
  task.env.ablate.random.hn_out= 0 \
  task.env.ablate.random.hn_in= 0 \
  task.env.ablate.random.cn_in= 0 \
  task.env.ablate.targeted_trial=false \
  task.env.ablate.targeted.obs_in= 0 \
  task.env.ablate.targeted.hn_out= 0 \
  task.env.ablate.targeted.hn_in= 0 \
  task.env.ablate.targeted.cn_in= 0 \
  task.env.ablate.ablations_obs_in=0 \
  task.env.ablate.ablations_hn_out=0 \
  task.env.ablate.ablations_hn_in=0 \
  task.env.ablate.ablations_cn_in=0 \
  task.env.output.export_data=True \
  task.env.evaluate.perturbPrescribed.perturbPrescribedOn=True \
  task.env.evaluate.perturbPrescribed.forceY=-3.0 \
  task.env.export_data_path=${export_path} \
  +output_path=${export_path}/${script_name}