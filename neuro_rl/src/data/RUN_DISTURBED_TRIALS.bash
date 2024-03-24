#!/bin/bash

# Define an array of model names
models_info=(
#   "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-BASELINE-01:last_AnymalTerrain_ep_1000_rew_20.962988.pth"
#   "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-DIST-01:last_AnymalTerrain_ep_5000_rew_16.480799.pth"
#   "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-TERR-01:last_AnymalTerrain_ep_2000_rew_18.73817.pth"
#   "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-DISTTERR-01:last_AnymalTerrain_ep_4600_rew_15.199695.pth"
#   "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-BASELINE-01:last_AnymalTerrain_ep_150_rew_8.168549.pth"
#   "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-DIST-01:last_AnymalTerrain_ep_4800_rew_20.043377.pth"
#   "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-TERR-01:last_AnymalTerrain_ep_1800_rew_18.174595.pth"
#   "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-DISTTERR-01:last_AnymalTerrain_ep_4800_rew_14.132425.pth"
#   "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-0.5MASS-LSTM16-TERR-01:last_AnymalTerrain_ep_3200_rew_21.073418.pth"
#   "A1Terrain_PPO_LSTM_NeuroRL:A1Terrain_NeuroRL_exp:A1-1.0MASS-LSTM16-TERR-01:last_A1Terrain_ep_4600_rew_16.256865.pth"

#   "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-FRONTIERSDISTTERR:last_AnymalTerrain_ep_1100_rew_14.392729.pth"
#   "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-FRONTIERSDISTTERR:last_AnymalTerrain_ep_2200_rew_19.53241.pth"
#   "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-FRONTIERSDISTTERR:last_AnymalTerrain_ep_3800_rew_20.310041.pth"
#   "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-FRONTIERSDISTTERR:last_AnymalTerrain_ep_3900_rew_20.14785.pth"
#   "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-FRONTIERSDISTTERR:last_AnymalTerrain_ep_4000_rew_20.387749.pth" 
#   "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-FRONTIERSDISTTERR:last_AnymalTerrain_ep_4100_rew_20.68903.pth"

  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-CORLDISTTERR:last_AnymalTerrain_ep_200_rew_6.8250656.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-CORLDISTTERR:last_AnymalTerrain_ep_300_rew_10.119753.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-CORLDISTTERR:last_AnymalTerrain_ep_400_rew_12.110974.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-CORLDISTTERR:last_AnymalTerrain_ep_500_rew_12.495365.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-CORLDISTTERR:last_AnymalTerrain_ep_1000_rew_14.850766.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-CORLDISTTERR:last_AnymalTerrain_ep_2000_rew_16.889687.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-CORLDISTTERR:last_AnymalTerrain_ep_3000_rew_16.622911.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-CORLDISTTERR:last_AnymalTerrain_ep_4000_rew_18.484495.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-CORLDISTTERR:last_AnymalTerrain_ep_5000_rew_16.690823.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-CORLDISTTERR:last_AnymalTerrain_ep_6000_rew_20.090017.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-CORLDISTTERR:last_AnymalTerrain_ep_6700_rew_20.21499.pth"

  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-CORLDISTTERR:last_AnymalTerrain_ep_200_rew_6.1486754.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-CORLDISTTERR:last_AnymalTerrain_ep_300_rew_8.433804.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-CORLDISTTERR:last_AnymalTerrain_ep_400_rew_10.192444.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-CORLDISTTERR:last_AnymalTerrain_ep_500_rew_11.477056.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-CORLDISTTERR:last_AnymalTerrain_ep_1000_rew_15.300709.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-CORLDISTTERR:last_AnymalTerrain_ep_1500_rew_15.248126.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-CORLDISTTERR:last_AnymalTerrain_ep_2000_rew_16.601225.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-CORLDISTTERR:last_AnymalTerrain_ep_2500_rew_16.594769.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-CORLDISTTERR:last_AnymalTerrain_ep_3000_rew_14.874878.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-CORLDISTTERR:last_AnymalTerrain_ep_3500_rew_17.787632.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-CORLDISTTERR:last_AnymalTerrain_ep_3700_rew_20.14857.pth"

  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-CORLTERR:last_AnymalTerrain_ep_200_rew_6.420168.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-CORLTERR:last_AnymalTerrain_ep_300_rew_8.896029.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-CORLTERR:last_AnymalTerrain_ep_400_rew_10.528543.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-CORLTERR:last_AnymalTerrain_ep_500_rew_13.228901.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-CORLTERR:last_AnymalTerrain_ep_1000_rew_14.604733.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-CORLTERR:last_AnymalTerrain_ep_1500_rew_14.298144.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-CORLTERR:last_AnymalTerrain_ep_2000_rew_18.007153.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-CORLTERR:last_AnymalTerrain_ep_2500_rew_18.825102.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-CORLTERR:last_AnymalTerrain_ep_3000_rew_19.434002.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-CORLTERR:last_AnymalTerrain_ep_3300_rew_20.003773.pth"

  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-CORLTERR:last_AnymalTerrain_ep_200_rew_5.884394.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-CORLTERR:last_AnymalTerrain_ep_300_rew_7.6767497.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-CORLTERR:last_AnymalTerrain_ep_400_rew_10.565976.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-CORLTERR:last_AnymalTerrain_ep_600_rew_12.610853.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-CORLTERR:last_AnymalTerrain_ep_1000_rew_14.291509.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-CORLTERR:last_AnymalTerrain_ep_1500_rew_14.035113.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-CORLTERR:last_AnymalTerrain_ep_2000_rew_16.989128.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-CORLTERR:last_AnymalTerrain_ep_2500_rew_17.63955.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-CORLTERR:last_AnymalTerrain_ep_3000_rew_18.42784.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-CORLTERR:last_AnymalTerrain_ep_3500_rew_18.885078.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-CORLTERR:last_AnymalTerrain_ep_3800_rew_20.163399.pth"

)

steps_after_stance_begins_values=0
length_s_values=(0.4 0.4 0.4 0.1 0.1 0.1 0.02 0.02 0.02)
forceY_values=(-0.333 -0.667 -1 -1 -2 -3 -4 -8 -12)

# length_s=0.02
# forceY_values=$(seq -12 2 12)

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
        num_envs=1 \
        task.env.specifiedCommandVelocityRanges.linear_x="[1, 1]" \
        task.env.specifiedCommandVelocityRanges.linear_y="[0, 0]" \
        task.env.specifiedCommandVelocityRanges.yaw_rate="[0, 0]" \
        task.env.specifiedCommandVelocityN.linear_x=1 \
        task.env.specifiedCommandVelocityN.linear_y=1 \
        task.env.specifiedCommandVelocityN.yaw_rate=1 \
        task.env.specifiedCommandVelocityN.n_copies=1 \
        task.env.export_data=true \
        task.env.export_data_actor=true \
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
        task.env.export_data_path=$export_path/$model_type/robustness_gradients_analysis/$steps_after_stance_begins/$length_s/$forceY/$model_name \
        +output_path=$export_path/$model_type/robustness_gradients_analysis/$steps_after_stance_begins/$length_s/$forceY/$model_name
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

    for i in ${!length_s_values[@]}; do
        length_s=${length_s_values[$i]}
        forceY=${forceY_values[$i]}

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


