#!/bin/bash

# Starting index for models
START_INDEX=256

# Number of models to train
N=1

# Extract the base name of the script
script_name=$(basename "$0")

# Optionally remove the .sh extension from the script name
script_name="${script_name%.*}"


# Change directory to where the train.py script is located
cd ../../../../IsaacGymEnvs/isaacgymenvs

seed=42

# Loop N times starting from START_INDEX
for ((i=START_INDEX; i<START_INDEX+N; i++)); do
  # Increment seed by 1 each iteration
  
  # Append an ID to train_dir and output_path
  id=$(printf "%02d" $i)
  export_path="../../neuro-rl/neuro_rl/models/"

  # Execute the python command with updated parameters
  python train.py task=AnymalTerrain_NeuroRL_train \
    train=AnymalTerrain_PPO_LSTM_NeuroRL \
    capture_video=False \
    capture_video_len=1000 \
    force_render=False \
    headless=True \
    wandb_activate=True \
    wandb_project=frontiers \
    wandb_entity=erush91 \
    task.env.terrain.terrainType=plane \
    task.env.urdfAsset.mass=urdf/anymal_c/urdf/anymal_minimal.urdf \
    task.env.urdfAsset.mass=53.16426 \
    task.env.learn.perturbRandom.perturbRandomOn=true \
    task.env.learn.perturbRandom.forceX=0.5 \
    task.env.learn.perturbRandom.forceY=0.5 \
    task.env.learn.perturbRandom.forceZ=0.5 \
    train.params.config.max_epochs=5000 \
    train.params.config.seq_length=4 \
    train.params.config.seq_len=4 \
    +output_path=${export_path}/${script_name}-${id} \
    +train_dir=${export_path} \
    +full_experiment_name=${script_name}-${id} \
    seed=${seed}
done