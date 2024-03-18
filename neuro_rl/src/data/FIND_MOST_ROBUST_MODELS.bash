#!/bin/bash

# This script goes and finds all models within the model folders given below, and (1) runs each model, 
# laterally disturbs each agent by the forceY (default -3.5) and step_after_stance_begins (default: 0), and records that in a csv file in the folder data/raw/find_most_robust_model/
# The specific subfolder structure is based on the specific parameters of the run: ${steps_after_stance_begins}/${forceY}/${model_name}

# Define an array of model names
models_info=(
    # "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-BASELINE-01" # -3.0, -3.5
    # "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-DIST-01" # -3.0, -3.5
    # "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-DISTTERR-01" # -3.0, -3.5

    # "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-BASELINE-01" # -3.0, -3.5
    # "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-DIST-01" # -3.0,
    # "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-TERR-01" # -3.0, -3.5 # see if there is any difference between these two in terms of FPs, config is identical
    # "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-DISTTERR-01" # -3.0, -3.5]

    # "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-0.5MASS-LSTM16-TERR-01" # -3.0 -3.5

    # "A1Terrain_PPO_LSTM_NeuroRL:A1Terrain_NeuroRL_exp:A1-1.0MASS-LSTM16-TERR-01" # -3.0, -3.5
    
    # "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-TERR-02" # -3.5 # see if there is any difference between these two in terms of FPs, config is identical
    # "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-TERR-201" TO DO # see if there is any difference between these two in terms of FPs, config is identical

    # "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-TERR-01-TERMINATION-1.0" # -3.0 (TERMINATION PENALTY DID NOT CREATE MONOTONIC ROBUSTNESS LIKE I HAD HOPED...)
    # "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-DIST-TERMINATION-1.0" # -3.0, -3.5 (TERMINATION PENALTY DID NOT CREATE MONOTONIC ROBUSTNESS LIKE I HAD HOPED...)

    # "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-TERR-THESIS" # -2.0, -3.0
    # "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-DISTTERR-01-THESIS" #

    # "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-BASELINE-01-CONDENSED"
    # "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-DIST-01-CONDENSED"
    # "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-TERR-01-CONDENSED"
    # "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-DISTTERR-01-CONDENSED"

    # "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-BASELINE-01-CONDENSED"
    # "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-DIST-01-CONDENSED"
    # "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-TERR-01-CONDENSED"
    # "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-DISTTERR-01-CONDENSED"

    "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-0.5MASS-LSTM16-TERR-01-CONDENSED"
    
    # "A1Terrain_PPO_LSTM_NeuroRL:A1Terrain_NeuroRL_exp:A1-1.0MASS-LSTM16-TERR-01-CONDENSED"
)

export_path="../../data/raw"

  # short disturbance
  declare -A short_disturbance=(
    [steps_after_stance_begins]=0
    [length_s]=0.02
    [forceY]=-12
  )

  # medium disturbance
  declare -A medium_disturbance=(
    [steps_after_stance_begins]=0
    [length_s]=0.1
    [forceY]=-3
  )

  # long disturbance
  declare -A long_disturbance=(
    [steps_after_stance_begins]=0
    [length_s]=0.4
    [forceY]=-1
  )

  # short disturbance
  declare -A short_smaller_disturbance=(
    [steps_after_stance_begins]=0
    [length_s]=0.02
    [forceY]=-8
  )

  # medium disturbance
  declare -A medium_smaller_disturbance=(
    [steps_after_stance_begins]=0
    [length_s]=0.1
    [forceY]=-2
  )

  # long disturbance
  declare -A long_smaller_disturbance=(
    [steps_after_stance_begins]=0
    [length_s]=0.4
    [forceY]=-0.667
  )

  # short disturbance
  declare -A short_smallest_disturbance=(
    [steps_after_stance_begins]=0
    [length_s]=0.02
    [forceY]=-4
  )

  # medium disturbance
  declare -A medium_smallest_disturbance=(
    [steps_after_stance_begins]=0
    [length_s]=0.1
    [forceY]=-1
  )

  # long disturbance
  declare -A long_smallest_disturbance=(
    [steps_after_stance_begins]=0
    [length_s]=0.4
    [forceY]=-0.333
  )

# Define a function that takes a file path as an argument
process_file() {
    local train_cfg_file="$1"
    local task_cfg_file="$2"
    local model_type="$3"
    local model_name="$4"
    local steps_after_stance_begins="$5"
    local length_s="$6"
    local forceY="$7"

    echo "------------------------------"
    echo "RUN PROCESSING:"
    echo "steps_after_stance_begins_values: $steps_after_stance_begins"
    echo "length_s: $length_s"
    echo "forceY_values: $forceY"
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
        num_envs=100 \
        task.env.specifiedCommandVelocityRanges.linear_x="[1, 1]" \
        task.env.specifiedCommandVelocityRanges.linear_y="[0, 0]" \
        task.env.specifiedCommandVelocityRanges.yaw_rate="[0, 0]" \
        task.env.specifiedCommandVelocityN.linear_x=1 \
        task.env.specifiedCommandVelocityN.linear_y=1 \
        task.env.specifiedCommandVelocityN.yaw_rate=1 \
        task.env.specifiedCommandVelocityN.n_copies=100 \
        task.env.export_data=false \
        task.env.export_data_actor=false \
        task.env.export_data_critic=false \
        task.env.evaluate.perturbPrescribed.perturbPrescribedOn=true \
        task.env.evaluate.perturbPrescribed.steps_after_stance_begins=$steps_after_stance_begins \
        task.env.evaluate.perturbPrescribed.length_s=$length_s \
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
        task.env.export_data_path=$export_path/$model_type/evaluate_robustness_throughout_training/$steps_after_stance_begins/$length_s/$forceY/$model_name \
        +output_path=$export_path/$model_type/evaluate_robustness_throughout_training/$steps_after_stance_begins/$length_s/$forceY/$model_name \
    )
}  

# Function to run the command with overridden parameters
run_command() {
    local train_cfg_file="$1"
    local task_cfg_file="$2"
    local model_type="$3"
    local steps_after_stance_begins="$4"
    local length_s="$5"
    local forceY="$6"

    search_dir="../../models/$model_type/nn" 

    find "$search_dir" -type f | sort -V | while read file; do
        model_name=$(basename "$file")

        echo "------------------------------"
        echo "MODEL PROCESSING:"
        echo "Train File: $train_cfg_file"
        echo "Task File: $task_cfg_file"
        echo "Model Type: $model_type"
        echo "Model Name: $model_name"
        echo "------------------------------"

        process_file "$train_cfg_file" "$task_cfg_file" "$model_type" "$model_name" "$steps_after_stance_begins_val" "$length_s_val" "$forceY"
    done
}


# Array of all runs
runs=(short_disturbance medium_disturbance long_disturbance short_smaller_disturbance medium_smaller_disturbance long_smaller_disturbance short_smallest_disturbance medium_smallest_disturbance long_smallest_disturbance)
# runs=(short_smallest_disturbance medium_smallest_disturbance long_smallest_disturbance)

# Loop over each run
for run in "${runs[@]}"; do
# Use 'declare -n' to create a nameref for easier access to associative array elements
declare -n current_run="$run"

# Execute Python command in a subshell with parameters from the current run
(

    steps_after_stance_begins_val="${current_run[steps_after_stance_begins]}"
    length_s_val="${current_run[length_s]}"
    forceY_val="${current_run[forceY]}"

    # Loop through the model names and call the sub-script for each one
    for model_info in "${models_info[@]}"; do
    
        echo "Running model: $model_info"

        # Split model_info into its components
        IFS=':' read -r train_cfg_file task_cfg_file model_type <<< "$model_info"

        run_command "$train_cfg_file" "$task_cfg_file" "$model_type" "$steps_after_stance_begins_val" "$length_s_val" "$forceY_val"

    done

)
done

# Wait for all background jobs to finish
wait


