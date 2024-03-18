#!/bin/bash

# Define an array of model names
models_info=(
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-BASELINE-01:last_AnymalTerrain_ep_1000_rew_20.962988.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-DIST-01:last_AnymalTerrain_ep_5000_rew_16.480799.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-TERR-01:last_AnymalTerrain_ep_2000_rew_18.73817.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-DISTTERR-01:last_AnymalTerrain_ep_4600_rew_15.199695.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-BASELINE-01:last_AnymalTerrain_ep_150_rew_8.168549.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-DIST-01:last_AnymalTerrain_ep_4800_rew_20.043377.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-TERR-01:last_AnymalTerrain_ep_1800_rew_18.174595.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-DISTTERR-01:last_AnymalTerrain_ep_4800_rew_14.132425.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-0.5MASS-LSTM16-TERR-01:last_AnymalTerrain_ep_3200_rew_21.073418.pth"
  "A1Terrain_PPO_LSTM_NeuroRL:A1Terrain_NeuroRL_exp:A1-1.0MASS-LSTM16-TERR-01:last_A1Terrain_ep_4600_rew_16.256865.pth"
)

steps_after_stance_begins_values=$(seq 0 1 20)
length_s=0.02
forceY_values=$(seq -12 2 12)

export_path="../../data/raw"

# Function to run the command with overridden parameters
run_command() {
    local train_cfg_file="$1"
    local task_cfg_file="$2"
    local model_type="$3"
    local model_name="$4"
    
    local steps_after_stance_begins="$5"
    local length_s="$6"
    local forceY="$7"

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
        task.env.export_data_path=$export_path/$model_type/final_model_robustness/$steps_after_stance_begins/$forceY/$model_name \
        +output_path=$export_path/$model_type/final_model_robustness/$steps_after_stance_begins/$forceY/$model_name
    )
}

# Loop through the model names and call the sub-script for each one
for model_info in "${models_info[@]}"; do
  
    echo "Running model: $model_info"

    # Split model_info into its components
    IFS=':' read -r train_cfg_file task_cfg_file model_type model_name <<< "$model_info"

    echo "------------------------------"
    echo "MODEL PROCESSING:"
    echo "Train File: $train_cfg_file"
    echo "Task File: $task_cfg_file"
    echo "Model Type: $model_type"
    echo "Model Name: $model_name"
    echo "------------------------------"

    for forceY in $forceY_values; do
        for steps_after_stance_begins in $steps_after_stance_begins_values; do
            echo "------------------------------"
            echo "RUN PROCESSING:"
            echo "steps_after_stance_begins_values: $steps_after_stance_begins"
            echo "length_s: $length_s"
            echo "forceY_values: $forceY"
            echo "------------------------------"
            run_command "$train_cfg_file" "$task_cfg_file" "$model_type" "$model_name" "$steps_after_stance_begins" "$length_s" "$forceY"
        done

    done

done

# Wait for all background jobs to finish
wait


