#!/bin/bash

# Define an array of model names
models_info=(
    # "A1Terrain_PPO_LSTM_NeuroRL:A1Terrain_NeuroRL_exp:A1-1.0MASS-LSTM16-TERR-01:last_A1Terrain_ep_13350_rew_19.288048.pth"
    # "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-0.5MASS-LSTM16-TERR-01:model_name"
    # "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-BASELINE-01:model_name"
    # "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-DIST-01:model_name"
    # "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-DISTTERR-01:last_AnymalTerrain_ep_5200_rew_15.612257.pth"
    # "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-BASELINE-01:last_AnymalTerrain_ep_950_rew_20.688946.pth"
    # "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-DIST-01:last_AnymalTerrain_ep_3500_rew_22.270075.pth"
    # "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-TERR-01:last_AnymalTerrain_ep_2550_rew_19.31447.pth" # see if there is any difference between these two in terms of FPs, config is identical
    # "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-TERR-02:last_AnymalTerrain_ep_3950_rew_18.507545.pth"
    # "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-TERR-201:model_name" # see if there is any difference between these two in terms of FPs, config is identical
    # "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-DISTTERR-01:last_AnymalTerrain_ep_3900_rew_18.605728.pth"
    # "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-TERR-01-TERMINATION-1.0:last_AnymalTerrain_ep_1800_rew_18.509405.pth"
    # "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-DIST-TERMINATION-1.0:last_AnymalTerrain_ep_900_rew_18.777004.pth"
)

forceY_values=-3.5 # $(seq -0.5 -0.5 -4)
steps_after_stance_begins_values=0

export_path="../../data/raw"

# Function to run the command with overridden parameters
run_command() {
    local train_cfg_file="$1"
    local task_cfg_file="$2"
    local model_type="$3"
    local model_name="$4"
    
    local steps_after_stance_begins="$5"
    local forceY="$6"

    # Execute Python command in a subshell with parameters from the current run
    (
    python ../../../../IsaacGymEnvs/isaacgymenvs/train.py \
        train=$train_cfg_file \
        task=$task_cfg_file \
        test=True \
        capture_video=False \
        capture_video_len=1000 \
        force_render=True \
        headless=False \
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
            echo "forceY_values: $forceY"
            echo "steps_after_stance_begins_values: $steps_after_stance_begins"
            echo "------------------------------"
            run_command "$train_cfg_file" "$task_cfg_file" "$model_type" "$model_name" "$steps_after_stance_begins" "$forceY"
        done

    done

done

# Wait for all background jobs to finish
wait


