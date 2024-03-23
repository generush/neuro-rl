#!/bin/bash

# Define an array of model names
models_info=(
  
  # "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-BASELINE-01:last_AnymalTerrain_ep_1000_rew_20.962988.pth"
  # "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-DIST-01:last_AnymalTerrain_ep_5000_rew_16.480799.pth"
  # "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-TERR-01:last_AnymalTerrain_ep_2000_rew_18.73817.pth"
  # "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-DISTTERR-01:last_AnymalTerrain_ep_4600_rew_15.199695.pth"

  # "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-BASELINE-01:last_AnymalTerrain_ep_150_rew_8.168549.pth"
  # "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-DIST-01:last_AnymalTerrain_ep_4800_rew_20.043377.pth"
  # "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-TERR-01:last_AnymalTerrain_ep_1800_rew_18.174595.pth"
  # "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-DISTTERR-01:last_AnymalTerrain_ep_4800_rew_14.132425.pth"

  # "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-0.5MASS-LSTM16-TERR-01:last_AnymalTerrain_ep_3200_rew_21.073418.pth"

  # "A1Terrain_PPO_LSTM_NeuroRL:A1Terrain_NeuroRL_exp:A1-1.0MASS-LSTM16-TERR-01:last_A1Terrain_ep_4600_rew_16.256865.pth"


  # "A1Terrain_PPO_LSTM_NeuroRL:A1Terrain_NeuroRL_exp:A1-1.0MASS-LSTM16-TERR-01:last_A1Terrain_ep_6550_rew_17.543756.pth"
  # "A1Terrain_PPO_LSTM_NeuroRL:A1Terrain_NeuroRL_exp:A1-1.0MASS-LSTM16-TERR-01:last_A1Terrain_ep_14000_rew_19.912346.pth"

  
  # "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-DISTTERR-01:last_AnymalTerrain_ep_1200_rew_12.890905.pth"
  

  # "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-FRONTIERSDISTTERR:last_AnymalTerrain_ep_1100_rew_14.392729.pth"
  # "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-FRONTIERSDISTTERR:last_AnymalTerrain_ep_2200_rew_19.53241.pth"
  # "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-FRONTIERSDISTTERR:last_AnymalTerrain_ep_3800_rew_20.310041.pth"
  # "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-FRONTIERSDISTTERR:last_AnymalTerrain_ep_3800_rew_20.310041.pth"
  # "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-FRONTIERSDISTTERR:last_AnymalTerrain_ep_3900_rew_20.14785.pth"
  # "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-FRONTIERSDISTTERR:last_AnymalTerrain_ep_4000_rew_20.387749.pth" 
  # "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-FRONTIERSDISTTERR:last_AnymalTerrain_ep_4100_rew_20.68903.pth"

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

export_path="../../data/raw"

# Loop through the model names and call the sub-script for each one
for model_info in "${models_info[@]}";
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

  # positive forward speeds (N=30x15) (A1 only)
  declare -A u_pos_30x15=(
    [num_envs]=450
    [linear_x_range]='[0.6, 1]'
    [linear_y_range]='[0, 0]'
    [yaw_rate_range]='[0, 0]'
    [linear_x_n]=30
    [linear_y_n]=1
    [yaw_rate_n]=1
    [n_copies]=15
  )

  # positive forward speeds (N=28x15)
  declare -A u_pos_28x15=(
    [num_envs]=420
    [linear_x_range]='[0.4, 1]'
    [linear_y_range]='[0, 0]'
    [yaw_rate_range]='[0, 0]'
    [linear_x_n]=28
    [linear_y_n]=1
    [yaw_rate_n]=1
    [n_copies]=15
  )

  # positive forward speeds (N=14x25)
  declare -A u_pos_14x25=(
    [num_envs]=350
    [linear_x_range]='[0.4, 1]'
    [linear_y_range]='[0, 0]'
    [yaw_rate_range]='[0, 0]'
    [linear_x_n]=14
    [linear_y_n]=1
    [yaw_rate_n]=1
    [n_copies]=25
  )

  # positive forward speeds (N=7x50)
  declare -A u_pos_7x50=(
    [num_envs]=350
    [linear_x_range]='[0.4, 1]'
    [linear_y_range]='[0, 0]'
    [yaw_rate_range]='[0, 0]'
    [linear_x_n]=7
    [linear_y_n]=1
    [yaw_rate_n]=1
    [n_copies]=50
  )

  # positive lateral speeds (N=7x50)
  declare -A v_pos_7x50=(
    [num_envs]=350
    [linear_x_range]='[0, 0]'
    [linear_y_range]='[0.4, 1]'
    [yaw_rate_range]='[0, 0]'
    [linear_x_n]=1
    [linear_y_n]=7
    [yaw_rate_n]=1
    [n_copies]=50
  )

  # negative+positive yaw rates (N=7x50)
  declare -A r_pos_7x50=(
    [num_envs]=350
    [linear_x_range]='[0, 0]'
    [linear_y_range]='[0, 0]'
    [yaw_rate_range]='[0, 1]'
    [linear_x_n]=1
    [linear_y_n]=1
    [yaw_rate_n]=7
    [n_copies]=50
  )

  # positive forward speed (N=100)
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

  # negative forward speed (N=100)
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

  # positive lateral speed (N=100)
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

  # negative lateral speed (N=100)
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

  # positive lateral speed (N=100)
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

  # negative lateral speed (N=100)
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

  # negative lateral speed (N=1)
  declare -A u_pos_1x1=(
    [num_envs]=1
    [linear_x_range]='[1, 1]'
    [linear_y_range]='[0, 0]'
    [yaw_rate_range]='[0, 0]'
    [linear_x_n]=1
    [linear_y_n]=1
    [yaw_rate_n]=1
    [n_copies]=1
  )

  # Add more runs as needed

  # Array of all runs
  # runs=(u_pos_28x15 u_pos_14x25 u_pos_7x50 v_pos_7x50 r_pos_7x50 u_pos_1x100 u_neg_1x100 v_pos_1x100 v_neg_1x100 r_pos_1x100 r_neg_1x100 u_pos_1x1)  # Add more run names as you define them
  # runs=(u_pos_30x15)  # Add more run names as you define them
  runs=(u_pos_28x15)  # Add more run names as you define them
  
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

      out_path="${export_path}/${model_type}/u_${linear_x_range_val}_${linear_x_n_val}_v_${linear_y_range_val}_${linear_y_n_val}_r_${yaw_rate_range_val}_${yaw_rate_n_val}_n_${n_copies_val}/${model_name}"

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