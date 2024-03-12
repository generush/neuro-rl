#!/bin/bash

# Define an array of model names
model_names=(
    "ANYMAL-1.0MASS-LSTM16-DISTTERR-01"
    "ANYMAL-1.0MASS-LSTM16-TERR-01"
    "ANYMAL-1.0MASS-LSTM16-DIST-01"
    "ANYMAL-1.0MASS-LSTM16-BASELINE-01"
)

export_path="../../data/raw"

# Function to run the command with overridden parameters
run_command() {
    local override_params="$1"
    local run_name="$2"  # This will be used to modify the output path dynamically
    # Execute Python command in a subshell with parameters from the current run
    (
      python ../../../../IsaacGymEnvs/isaacgymenvs/train.py task=AnymalTerrain_NeuroRL_exp \
        train=AnymalTerrain_PPO_LSTM_NeuroRL \
        capture_video=False \
        capture_video_len=1000 \
        force_render=True \
        headless=False \
        test=True \
        checkpoint=../../models/${model_name}/nn/model.pth \
        num_envs=400 \
        task.env.specifiedCommandVelocityRanges.linear_x=[1, 1] \
        task.env.specifiedCommandVelocityRanges.linear_y=[0, 0] \
        task.env.specifiedCommandVelocityRanges.yaw_rate=[0, 0] \
        task.env.specifiedCommandVelocityN.linear_x=1 \
        task.env.specifiedCommandVelocityN.linear_y=1 \
        task.env.specifiedCommandVelocityN.yaw_rate=1 \
        task.env.specifiedCommandVelocityN.n_copies=400 \
        task.env.export_data=true \
        task.env.export_data_actor=true \
        task.env.export_data_critic=false \
        task.env.evaluate.perturbPrescribed.perturbPrescribedOn=true \
        task.env.evaluate.perturbPrescribed.steps_after_stance_begins: 0 \
        task.env.evaluate.perturbPrescribed.forceY=-3.5 \
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
        task.env.export_data_path=${export_path}/${model_name}/${run_name} \
        +output_path=${export_path}/${model_name}/${run_name}
    )
}
done

forceY_values=$(seq -4 0.5 4)
steps_after_stance_begins_values=$(seq 0 1 20)

# Loop through the model names
for model_name in "${model_names[@]}"
do
    echo "Running model: $model_name"

    for forceY in $forceY_values; do
        for steps_after_stance_begins in steps_after_stance_begins_values; do
            run_command "task.env.evaluate.perturbPrescribed.forceY=$forceY" "task.env.evaluate.perturbPrescribed.steps_after_stance_begins=$steps_after_stance_begins" "RANDOM-ABLATION-TRIAL-hn-out-${forceY}"
    done
done

# Wait for all background jobs to finish
wait