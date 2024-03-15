#!/bin/bash

# This script goes and finds all models within the model folders given below, and (1) runs each model, 
# laterally disturbs each agent by the forceY (default -3.5) and step_after_stance_begins (default: 0), and records that in a csv file in the folder data/raw/find_most_robust_model/
# The specific subfolder structure is based on the specific parameters of the run: ${steps_after_stance_begins}/${forceY}/${model_name}

# Define an array of model names
models_info=(
    "A1Terrain_PPO_LSTM_NeuroRL:A1Terrain_NeuroRL_exp:A1-1.0MASS-LSTM16-TERR-01"
    # "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-0.5MASS-LSTM16-TERR-01"
    # "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-BASELINE-01"
    # "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-DIST-01"
    # # "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-TERR-01" tb
    # "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-DISTTERR-01"
    # "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-BASELINE-01"
    # "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-DIST-01"
    # "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-TERR-01" # see if there is any difference between these two in terms of FPs, config is identical
    # "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-TERR-201" # see if there is any difference between these two in terms of FPs, config is identical
    # "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-DISTTERR-01" DONE
)

export_path="../../data/raw"

# Define a function that takes a file path as an argument
process_file() {
    local train_cfg_file="$1"
    local task_cfg_file="$2"
    local model_type="$3"
    local model_name="$4"
    
    local forceY=-3.5
    local steps_after_stance_begins=0

    echo "------------------------------"
    echo "RUN PROCESSING:"
    echo "forceY_values: $forceY"
    echo "steps_after_stance_begins_values: $steps_after_stance_begins"
    echo "------------------------------"

    # Execute Python command in a subshell with parameters from the current run
    (
    python ../../../../IsaacGymEnvs/isaacgymenvs/train.py \
        train=$train_cfg_file \
        task=$task_cfg_file \
        test=True \
        capture_video=False \
        capture_video_len=1000 \
        force_render=False \
        headless=True \
        checkpoint=../../models/$model_type/nn/$model_name \
        num_envs=400 \
        task.env.specifiedCommandVelocityRanges.linear_x="[1, 1]" \
        task.env.specifiedCommandVelocityRanges.linear_y="[0, 0]" \
        task.env.specifiedCommandVelocityRanges.yaw_rate="[0, 0]" \
        task.env.specifiedCommandVelocityN.linear_x=1 \
        task.env.specifiedCommandVelocityN.linear_y=1 \
        task.env.specifiedCommandVelocityN.yaw_rate=1 \
        task.env.specifiedCommandVelocityN.n_copies=400 \
        task.env.export_data=false \
        task.env.export_data_actor=false \
        task.env.export_data_critic=false \
        task.env.evaluate.perturbPrescribed.perturbPrescribedOn=true \
        task.env.evaluate.perturbPrescribed.steps_after_stance_begins=$steps_after_stance_begins \
        task.env.evaluate.perturbPrescribed.forceY=$forceY \
        task.env.ablate.wait_until_disturbance=false \
        task.env.ablate.random_trial=false \
        task.env.ablate.random.obs_in=0 \
        task.env.ablate.random.hn_out=0 \
        task.env.ablate.random.hn_in=0 \
        task.env.ablate.random.cn_in=0 \
        task.env.ablate.targeted_trial=false \
        task.env.ablate.targeted.obs_in=0 \
        task.env.ablate.targeted.hn_out=0 \
        task.env.ablate.targeted.hn_in=0 \
        task.env.ablate.targeted.cn_in=0 \
        task.env.export_data_path=$export_path/$model_type/find_most_robust_model/$steps_after_stance_begins/$forceY/$model_name \
        +output_path=$export_path/$model_type/find_most_robust_model/$steps_after_stance_begins/$forceY/$model_name \
    )
}  

# Function to run the command with overridden parameters
run_command() {
    local train_cfg_file="$1"
    local task_cfg_file="$2"
    local model_type="$3"

    search_dir="../../models/$model_type/nn" 

    find "$search_dir" -type f | sort -V | while read file; do
        model_name=$(basename "$file")
        process_file "$train_cfg_file" "$task_cfg_file" "$model_type" "$model_name"
    done
}

# Loop through the model names and call the sub-script for each one
for model_info in "${models_info[@]}"; do
  
    echo "Running model: $model_info"

    # Split model_info into its components
    IFS=':' read -r train_cfg_file task_cfg_file model_type <<< "$model_info"

    echo "------------------------------"
    echo "MODEL PROCESSING:"
    echo "Train File: $train_cfg_file"
    echo "Task File: $task_cfg_file"
    echo "Model Type: $model_type"
    echo "------------------------------"

    run_command "$train_cfg_file" "$task_cfg_file" "$model_type"

done

# Wait for all background jobs to finish
wait


