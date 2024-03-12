#!/bin/bash

# Get the model name from the first argument
model_name="$1"

# Check if model name is provided
if [ -z "$model_name" ]; then
    echo "Error: Model name not provided."
    exit 1
fi

export_path="../../data/raw"

# Define an associative array for each run with specific parameters

# positive forward speeds (N=14x20)
declare -A run1=(
  [num_envs]=280
  [linear_x_range]='[0.4, 1]'
  [linear_y_range]='[0, 0]'
  [yaw_rate_range]='[0, 0]'
  [linear_x_n]=14
  [linear_y_n]=1
  [yaw_rate_n]=1
  [n_copies]=20
)

# positive forward speeds (N=7x40)
declare -A run2=(
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
declare -A run3=(
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
declare -A run4=(
  [num_envs]=280
  [linear_x_range]='[1, 1]'
  [linear_y_range]='[0, 0]'
  [yaw_rate_range]='[-1, 1]'
  [linear_x_n]=1
  [linear_y_n]=1
  [yaw_rate_n]=7
  [n_copies]=40
)

# positive forward speed (N=1)
declare -A run5=(
  [num_envs]=1
  [linear_x_range]='[1, 1]'
  [linear_y_range]='[0, 0]'
  [yaw_rate_range]='[0, 0]'
  [linear_x_n]=1
  [linear_y_n]=1
  [yaw_rate_n]=1
  [n_copies]=1
)

# negative forward speed (N=1)
declare -A run6=(
  [num_envs]=1
  [linear_x_range]='[-1, -1]'
  [linear_y_range]='[0, 0]'
  [yaw_rate_range]='[0, 0]'
  [linear_x_n]=1
  [linear_y_n]=1
  [yaw_rate_n]=1
  [n_copies]=1
)

# positive lateral speed (N=1)
declare -A run7=(
  [num_envs]=1
  [linear_x_range]='[0, 0]'
  [linear_y_range]='[1, 1]'
  [yaw_rate_range]='[0, 0]'
  [linear_x_n]=1
  [linear_y_n]=1
  [yaw_rate_n]=1
  [n_copies]=1
)

# negative lateral speed (N=1)
declare -A run8=(
  [num_envs]=1
  [linear_x_range]='[0, 0]'
  [linear_y_range]='[-1, -1]'
  [yaw_rate_range]='[0, 0]'
  [linear_x_n]=1
  [linear_y_n]=1
  [yaw_rate_n]=1
  [n_copies]=1
)

# positive lateral speed (N=1)
declare -A run9=(
  [num_envs]=1
  [linear_x_range]='[1, 1]'
  [linear_y_range]='[0, 0]'
  [yaw_rate_range]='[1, 1]'
  [linear_x_n]=1
  [linear_y_n]=1
  [yaw_rate_n]=1
  [n_copies]=1
)

# negative lateral speed (N=1)
declare -A run10=(
  [num_envs]=1
  [linear_x_range]='[1, 1]'
  [linear_y_range]='[0, 0]'
  [yaw_rate_range]='[-1, -1]'
  [linear_x_n]=1
  [linear_y_n]=1
  [yaw_rate_n]=1
  [n_copies]=1
)

# Add more runs as needed

# Array of all runs
# runs=(run1)  # Add more run names as you define them
runs=(run1 run2 run3 run4 run5 run6 run7 run8 run9 run10)  # Add more run names as you define them
runs=(run1)  # Add more run names as you define them

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

    out_path="${export_path}/${model_name}/u_${linear_x_range_val}_${linear_x_n_val}_v_${linear_y_range_val}_${linear_y_n_val}_r_${yaw_rate_range_val}_${yaw_rate_n_val}_n_${n_copies_val}"

    python ../../../../IsaacGymEnvs/isaacgymenvs/train.py task=A1Terrain_NeuroRL_exp \
      train=A1Terrain_PPO_LSTM_NeuroRL \
      capture_video=False \
      capture_video_len=1000 \
      force_render=True \
      headless=False \
      test=True \
      checkpoint=../../models/${model_name}/nn/model.pth \
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
