#!/bin/bash

# This script goes and finds all models within the model folders given below, and (1) runs each model, 
# laterally disturbs each agent by the forceY (default -3.5) and step_after_stance_begins (default: 0), and records that in a csv file in the folder data/raw/find_most_robust_model/
# The specific subfolder structure is based on the specific parameters of the run: ${steps_after_stance_begins}/${forceY}/${model_name}

# Define an array of model names
models_info=(

    # "AnymalTerrainPPO_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-FF-CORLTERR" # AnymalTerrain_05-12-53-48/nn/AnymalTerrain.pth
    # "AnymalTerrainPPO_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-FF-CORLDISTTERR" # AnymalTerrain_04-01-13-21/nn/last_AnymalTerrain_ep_19250_rew_20.038643.pth
    # "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-CORLTERR" # AnymalTerrain_06-17-40-50/nn/last_AnymalTerrain_ep_3300_rew_20.003773.pth
    # "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-CORLTERR" # AnymalTerrain_07-03-29-04/nn/last_AnymalTerrain_ep_3800_rew_20.163399.pth
    # "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-CORLDISTTERR" # runs/AnymalTerrain_06-00-14-59/nn/last_AnymalTerrain_ep_6700_rew_20.21499.pth
    # "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-CORLDISTTERR" # runs/AnymalTerrain_04-15-37-26/nn/last_AnymalTerrain_ep_3700_rew_20.14857.pth
    # "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-CORLDISTTERR2" # AnymalTerrain_08-04-24-44/nn/last_AnymalTerrain_ep_4500_rew_20.877975.pth

    # "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-FRONTIERSDISTTERR" # AnymalTerrain_2023-08-24_15-24-12/nn/last_AnymalTerrain_ep_3200_rew_20.145746.pth
    # "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-FRONTIERSTERR" # AnymalTerrain_2023-08-27_17-23-34/AnymalTerrain/nn/last_AnymalTerrain_ep_2900_rew_20.2482.pth
    # "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-FRONTIERSDIST" # AnymalTerrain_2023-08-24_14-17-13/nn/last_AnymalTerrain_ep_900_rew_20.139568.pth
    # "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-FRONTIERSBASELINE" # 2023-09-13-18-33_AnymalTerrain/nn/last_AnymalTerrain_ep_700_rew_20.361492.pth

    # "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-DISTTERR-01-CONDENSED" # ANYMAL-1.0MASS-LSTM4-DISTTERR-01/nn/last_AnymalTerrain_ep_4800_rew_14.132425.pth
    # "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-TERR-01-CONDENSED" # ANYMAL-1.0MASS-LSTM4-TERR-01/nn/last_AnymalTerrain_ep_1800_rew_18.174595.pth
    # "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-DIST-01-CONDENSED" # ANYMAL-1.0MASS-LSTM4-DIST-01/nn/last_AnymalTerrain_ep_4800_rew_20.043377.pth
    # "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-BASELINE-01-CONDENSED" # ANYMAL-1.0MASS-LSTM4-BASELINE-01/nn/last_AnymalTerrain_ep_150_rew_8.168549.pth

    # "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-DISTTERR-01-CONDENSED" # ANYMAL-1.0MASS-LSTM16-DISTTERR-01/nn/last_AnymalTerrain_ep_4600_rew_15.199695.pth
    # "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-TERR-01-CONDENSED" # ANYMAL-1.0MASS-LSTM16-TERR-01/nn/last_AnymalTerrain_ep_2000_rew_18.73817.pth
    # "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-DIST-01-CONDENSED" # ANYMAL-1.0MASS-LSTM16-DIST-01/nn/last_AnymalTerrain_ep_5000_rew_16.480799.pth
    # "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-BASELINE-01-CONDENSED" # ANYMAL-1.0MASS-LSTM16-BASELINE-01/nn/last_AnymalTerrain_ep_1000_rew_20.962988.pth

    # "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-0.5MASS-LSTM16-TERR-01-CONDENSED" # ANYMAL-0.5MASS-LSTM16-TERR-01/nn/last_AnymalTerrain_ep_3200_rew_21.073418.pth
    
    # "A1Terrain_PPO_LSTM_NeuroRL:A1Terrain_NeuroRL_exp:A1-1.0MASS-LSTM16-TERR-01-CONDENSED" # A1-1.0MASS-LSTM16-TERR-01/nn/last_A1Terrain_ep_4600_rew_16.256865.pth
    # "A1Terrain_PPO_LSTM_NeuroRL:A1Terrain_NeuroRL_exp:A1-1.0MASS-LSTM16-TERR-01-CONDENSEDLONG" # A1-1.0MASS-LSTM16-TERR-01/nn/last_A1Terrain_ep_4600_rew_16.256865.pth

    "A1Terrain_PPO_LSTM_NeuroRL:A1Terrain_NeuroRL_exp:A1-1.0MASS-LSTM16-DISTTERR-01-CONDENSED" # A1-1.0MASS-LSTM16-TERR-01/nn/last_A1Terrain_ep_4600_rew_16.256865.pth

    # COULD REVISIT DOING THE UNCONDENSED MODELS FOR THE THESIS MODELS

    # "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-DISTTERR-01" # ANYMAL-1.0MASS-LSTM4-DISTTERR-01/nn/last_AnymalTerrain_ep_4800_rew_14.132425.pth
    # "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-TERR-01" # ANYMAL-1.0MASS-LSTM4-TERR-01/nn/last_AnymalTerrain_ep_1800_rew_18.174595.pth
    # "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-DIST-01" # ANYMAL-1.0MASS-LSTM4-DIST-01/nn/last_AnymalTerrain_ep_4800_rew_20.043377.pth
    # "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-BASELINE-01" # ANYMAL-1.0MASS-LSTM4-BASELINE-01/nn/last_AnymalTerrain_ep_150_rew_8.168549.pth

    # "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-DISTTERR-01" # ANYMAL-1.0MASS-LSTM16-DISTTERR-01/nn/last_AnymalTerrain_ep_4600_rew_15.199695.pth
    # "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-TERR-01" # ANYMAL-1.0MASS-LSTM16-TERR-01/nn/last_AnymalTerrain_ep_2000_rew_18.73817.pth
    # "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-DIST-01" # ANYMAL-1.0MASS-LSTM16-DIST-01/nn/last_AnymalTerrain_ep_5000_rew_16.480799.pth
    # "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-BASELINE-01" # ANYMAL-1.0MASS-LSTM16-BASELINE-01/nn/last_AnymalTerrain_ep_1000_rew_20.962988.pth    # "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-0.5MASS-LSTM16-TERR-01-CONDENSED" # ANYMAL-0.5MASS-LSTM16-TERR-01/nn/last_AnymalTerrain_ep_3200_rew_21.073418.pth
    
    # "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-0.5MASS-LSTM16-TERR-01-CONDENSED" # ANYMAL-0.5MASS-LSTM16-TERR-01/nn/last_AnymalTerrain_ep_3200_rew_21.073418.pth
    # "A1Terrain_PPO_LSTM_NeuroRL:A1Terrain_NeuroRL_exp:A1-1.0MASS-LSTM16-TERR-01" # A1-1.0MASS-LSTM16-TERR-01/nn/last_A1Terrain_ep_4600_rew_16.256865.pth
    # "A1Terrain_PPO_LSTM_NeuroRL:A1Terrain_NeuroRL_exp:A1-1.0MASS-LSTM16-DISTTERR-01" # A1-1.0MASS-LSTM16-TERR-01/nn/last_A1Terrain_ep_4600_rew_16.256865.pth



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
# runs=(short_disturbance medium_disturbance long_disturbance)

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


