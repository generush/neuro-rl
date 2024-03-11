#!/bin/bash

# Get the model name from the first argument
model_name="$1"

# Check if model name is provided
if [ -z "$model_name" ]; then
    echo "Error: Model name not provided."
    exit 1
fi

export_path="../../data/raw"

# Add more runs as needed

# Array of all runs
# runs=(run1)  # Add more run names as you define them
runs=(run1 run2 run3 run4 run5 run6 run7 run8 run9 run10)  # Add more run names as you define them

# Loop over each run
for run in "${runs[@]}"; do
  # Use 'declare -n' to create a nameref for easier access to associative array elements
  declare -n current_run="$run"

  # Execute Python command in a subshell with parameters from the current run
  (
    # Custom setup for the subshell (if needed)
    
    # random_trial_val="${current_run[random_trial]}"
    
    model_path="${export_path}/${model_name}" \
    # run_name="${}"
    # run_path="u_${linear_x_range_val}_${linear_x_n_val}_v_${linear_y_range_val}_${linear_y_n_val}_r_${yaw_rate_range_val}_${yaw_rate_n_val}_n_${n_copies_val}"

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
      task.env.evaluate.perturbPrescribed.forceY=-3.5 \
      task.env.ablate.wait_until_disturbance=false \
      task.env.ablate.scl_pca_path=../../neuro-rl/neuro_rl/data/processed/${model_name}/u_0.4_1.0_14_v_0._0._1_r_0._0._1_n_20 \
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
      task.env.export_data_path=${model_path}_PERTURB \
      +output_path=${model_path}

                "task.env.export_data_path=../../neuro-rl/neuro_rl/data/processed/ANYMAL-1.0MASS-LSTM16-DISTTERR-01/u_0.4_1.0_14_v_0._0._1_r_0._0._1_n_20_perturb",
                "+output_path=../../neuro-rl/neuro_rl/data/processed/ANYMAL-1.0MASS-LSTM16-DISTTERR-01/u_0.4_1.0_14_v_0._0._1_r_0._0._1_n_20_perturb",
  )
done

# Wait for all background jobs to finish
wait
