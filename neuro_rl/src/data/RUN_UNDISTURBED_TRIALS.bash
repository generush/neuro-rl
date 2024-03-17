#!/bin/bash

# Define an array of model names
models_info=(
  # "A1Terrain_PPO_LSTM_NeuroRL:A1Terrain_NeuroRL_exp:A1-1.0MASS-LSTM16-TERR-01:last_A1Terrain_ep_13350_rew_19.288048.pth"
  # "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-0.5MASS-LSTM16-TERR-01:model"
  # "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-BASELINE-01:model"
  # "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-DIST-01:model"
  # "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-TERR-01:model"
  # "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-DISTTERR-01:last_AnymalTerrain_ep_5200_rew_15.612257.pth"
  # "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-BASELINE-01:model"
  # "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-DIST-01:model:last_AnymalTerrain_ep_3500_rew_22.270075.pth"
  # "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-TERR-01:last_AnymalTerrain_ep_2550_rew_19.31447.pth" # see if there is any difference between these two in terms of FPs, config is identical
  # "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-TERR-201:model" # see if there is any difference between these two in terms of FPs, config is identical
  # "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-DISTTERR-01:last_AnymalTerrain_ep_3900_rew_18.605728.pth"
)

export_path="../../data/raw"

# Loop through the model names and call the sub-script for each one
for model_info in "${models_info[@]}"
do
   
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

  # Define an associative array for each run with specific parameters

  # positive forward speeds (N=14x20)
  declare -A u_pos_28x10=(
    [num_envs]=280
    [linear_x_range]='[0.4, 1]'
    [linear_y_range]='[0, 0]'
    [yaw_rate_range]='[0, 0]'
    [linear_x_n]=28
    [linear_y_n]=1
    [yaw_rate_n]=1
    [n_copies]=10
  )

  # positive forward speeds (N=7x40)
  declare -A u_pos_7x40=(
    [num_envs]=280
    [linear_x_range]='[0.4, 1]'
    [linear_y_range]='[0, 0]'
    [yaw_rate_range]='[0, 0]'
    [linear_x_n]=7
    [linear_y_n]=1
    [yaw_rate_n]=1
    [n_copies]=40
  )

  # positive lateral speeds (N=7x40)
  declare -A v_pos_7x40=(
    [num_envs]=280
    [linear_x_range]='[0, 0]'
    [linear_y_range]='[0.4, 1]'
    [yaw_rate_range]='[0, 0]'
    [linear_x_n]=1
    [linear_y_n]=7
    [yaw_rate_n]=1
    [n_copies]=40
  )

  # negative+positive yaw rates (N=7x40)
  declare -A r_pos_7x40=(
    [num_envs]=280
    [linear_x_range]='[0, 0]'
    [linear_y_range]='[0, 0]'
    [yaw_rate_range]='[0, 1]'
    [linear_x_n]=1
    [linear_y_n]=1
    [yaw_rate_n]=7
    [n_copies]=40
  )

  # positive forward speed (N=1)
  declare -A u_pos_1x100=(
    [num_envs]=100
    [linear_x_range]='[1, 1]'
    [linear_y_range]='[0, 0]'
    [yaw_rate_range]='[0, 0]'
    [linear_x_n]=1
    [linear_y_n]=1
    [yaw_rate_n]=1
    [n_copies]=100
  )

  # negative forward speed (N=1)
  declare -A u_neg_1x100=(
    [num_envs]=100
    [linear_x_range]='[-1, -1]'
    [linear_y_range]='[0, 0]'
    [yaw_rate_range]='[0, 0]'
    [linear_x_n]=1
    [linear_y_n]=1
    [yaw_rate_n]=1
    [n_copies]=100
  )

  # positive lateral speed (N=1)
  declare -A v_pos_1x100=(
    [num_envs]=100
    [linear_x_range]='[0, 0]'
    [linear_y_range]='[1, 1]'
    [yaw_rate_range]='[0, 0]'
    [linear_x_n]=1
    [linear_y_n]=1
    [yaw_rate_n]=1
    [n_copies]=100
  )

  # negative lateral speed (N=1)
  declare -A v_neg_1x100=(
    [num_envs]=100
    [linear_x_range]='[0, 0]'
    [linear_y_range]='[-1, -1]'
    [yaw_rate_range]='[0, 0]'
    [linear_x_n]=1
    [linear_y_n]=1
    [yaw_rate_n]=1
    [n_copies]=100
  )

  # positive lateral speed (N=1)
  declare -A r_pos_1x100=(
    [num_envs]=100
    [linear_x_range]='[1, 1]'
    [linear_y_range]='[0, 0]'
    [yaw_rate_range]='[1, 1]'
    [linear_x_n]=1
    [linear_y_n]=1
    [yaw_rate_n]=1
    [n_copies]=100
  )

  # negative lateral speed (N=1)
  declare -A r_neg_1x100=(
    [num_envs]=100
    [linear_x_range]='[1, 1]'
    [linear_y_range]='[0, 0]'
    [yaw_rate_range]='[-1, -1]'
    [linear_x_n]=1
    [linear_y_n]=1
    [yaw_rate_n]=1
    [n_copies]=100
  )

  # Add more runs as needed

  # Array of all runs
  runs=(u_pos_28x10 u_pos_7x40 v_pos_7x40 r_pos_7x40 u_pos_1x100 u_neg_1x100 v_pos_1x100 v_neg_1x100 r_pos_1x100 r_neg_1x100)  # Add more run names as you define them
  
  # Loop over each run
  for run in "${runs[@]}"; do
    # Use 'declare -n' to create a nameref for easier access to associative array elements
    declare -n current_run="$run"

    # Execute Python command in a subshell with parameters from the current run
    (
      # Custom setup for the subshell (if needed)
        
      linear_x_range_val="${current_run[linear_x_range]#[}"
      linear_x_range_val="${linear_x_range_val%]}"
      linear_x_range_val="${linear_x_range_val//,/_}"
      linear_x_range_val="${linear_x_range_val// /}"
      
      linear_y_range_val="${current_run[linear_y_range]#[}"
      linear_y_range_val="${linear_y_range_val%]}"
      linear_y_range_val="${linear_y_range_val//,/_}"
      linear_y_range_val="${linear_y_range_val// /}"
      
      yaw_rate_range_val="${current_run[yaw_rate_range]#[}"
      yaw_rate_range_val="${yaw_rate_range_val%]}"
      yaw_rate_range_val="${yaw_rate_range_val//,/_}"
      yaw_rate_range_val="${yaw_rate_range_val// /}"

      linear_x_n_val="${current_run[linear_x_n]}"
      linear_y_n_val="${current_run[linear_y_n]}"
      yaw_rate_n_val="${current_run[yaw_rate_n]}"
      n_copies_val="${current_run[n_copies]}"

      out_path="${export_path}/${model_type}/u_${linear_x_range_val}_${linear_x_n_val}_v_${linear_y_range_val}_${linear_y_n_val}_r_${yaw_rate_range_val}_${yaw_rate_n_val}_n_${n_copies_val}"

      python ../../../../IsaacGymEnvs/isaacgymenvs/train.py \
        task=${task_cfg_file} \
        train=${train_cfg_file}\
        test=True \
        capture_video=False \
        capture_video_len=1000 \
        force_render=False \
        headless=True \
        checkpoint=../../models/${model_type}/nn/$model_name \
        num_envs=${current_run[num_envs]} \
        task.env.specifiedCommandVelocityRanges.linear_x="${current_run[linear_x_range]}" \
        task.env.specifiedCommandVelocityRanges.linear_y="${current_run[linear_y_range]}" \
        task.env.specifiedCommandVelocityRanges.yaw_rate="${current_run[yaw_rate_range]}" \
        task.env.specifiedCommandVelocityN.linear_x=${current_run[linear_x_n]} \
        task.env.specifiedCommandVelocityN.linear_y=${current_run[linear_y_n]} \
        task.env.specifiedCommandVelocityN.yaw_rate=${current_run[yaw_rate_n]} \
        task.env.specifiedCommandVelocityN.n_copies=${current_run[n_copies]} \
        task.env.export_data=true \
        task.env.export_data_actor=true \
        task.env.export_data_critic=false \
        task.env.export_data_path=${out_path} \
        +output_path="${out_path}"
    )
  done

  # Wait for all background jobs to finish
  wait

done