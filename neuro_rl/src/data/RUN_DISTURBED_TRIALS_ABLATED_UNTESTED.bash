#!/bin/bash

# Define an array of model names
models_info=(
    # "A1Terrain_PPO_LSTM_NeuroRL:A1Terrain_NeuroRL_exp:A1-1.0MASS-LSTM16-TERR-01:model_name" # need to rerun training for longer epochs (model did not yet converge)
    # "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-0.5MASS-LSTM16-TERR-01:model_name"
    # "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-BASELINE-01:model_name"
    # "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-DIST-01:model_name"
    # "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-TERR-01:model_name"
    # "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-DISTTERR-01:model_name"
    # "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-BASELINE-01:model_name"
    # "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-DIST-01:model_name"
    # "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-TERR-01:model_name" # see if there is any difference between these two in terms of FPs, config is identical
    # "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-TERR-201:model_name" # see if there is any difference between these two in terms of FPs, config is identical
    "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-DISTTERR-01:model_name"
)

forceY_values=$(seq -0.5 -0.5 -4)
steps_after_stance_begins_values=0

export_path="../../data/raw"

# Function to run the command with overridden parameters
run_command() {
    local train_cfg_file="$1"
    local task_cfg_file="$2"
    local model_type="$3"
    local model_name="$4"

    local override_params="$1"
    local run_name="$2"  # This will be used to modify the output path dynamically
    # Execute Python command in a subshell with parameters from the current run
    
    local steps_after_stance_begins=0
    local forceY=0

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
        task.env.export_data_path=$export_path/$model_type/RUN_DISTURBED_ABLATED_TRIALS/$steps_after_stance_begins/$forceY/$model_name \
        +output_path=$export_path/$model_type/RUN_DISTURBED_ABLATED_TRIALS/$steps_after_stance_begins/$forceY/$model_name
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

    for hn_out in $hn_out_values; do
        run_command "$train_cfg_file" "$task_cfg_file" "$model_type" "$model_name" \
        "task.env.ablate.random_trial=true task.env.ablate.random.hn_out=$hn_out" "RANDOM-ABLATION-TRIAL-hn-out-${hn_out}"
    done

    for hn_in in $hn_in_values; do
        run_command "$train_cfg_file" "$task_cfg_file" "$model_type" "$model_name" \
        "task.env.ablate.random_trial=true task.env.ablate.random.hn_in=$hn_in" "RANDOM-ABLATION-TRIAL-hn-in-${hn_in}"
    done

    for cn_in in $cn_in_values; do
        run_command "$train_cfg_file" "$task_cfg_file" "$model_type" "$model_name" \
        "task.env.ablate.random_trial=true task.env.ablate.random.cn_in=$cn_in" "RANDOM-ABLATION-TRIAL-cn-in-${cn_in}"
    done

    for hn_out in $hn_out_values; do
        run_command "$train_cfg_file" "$task_cfg_file" "$model_type" "$model_name" \
        "task.env.ablate.targeted_trial=true task.env.ablate.random.hn_out=$hn_out" "TARGETED-ABLATION-TRIAL-hn-out-${hn_out}"
    done

    for hn_in in $hn_in_values; do
        run_command "$train_cfg_file" "$task_cfg_file" "$model_type" "$model_name" \
        "task.env.ablate.targeted_trial=true task.env.ablate.random.hn_in=$hn_in" "TARGETED-ABLATION-TRIAL-hn-in-${hn_in}"
    done

    for cn_in in $cn_in_values; do
        run_command "$train_cfg_file" "$task_cfg_file" "$model_type" "$model_name" \
        "task.env.ablate.targeted_trial=true task.env.ablate.random.cn_in=$cn_in" "TARGETED-ABLATION-TRIAL-cn-in-${cn_in}"
    done

done