#!/bin/bash

# Define an array of model names
models_info=(

#   "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-BASELINE-01:last_AnymalTerrain_ep_1000_rew_20.962988.pth" # DONE
#   "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-DIST-01:last_AnymalTerrain_ep_5000_rew_16.480799.pth" # DONE
#   "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-TERR-01:last_AnymalTerrain_ep_2000_rew_18.73817.pth" # DONE
#   "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-DISTTERR-01:last_AnymalTerrain_ep_4600_rew_15.199695.pth" # DONE
#   "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-BASELINE-01:last_AnymalTerrain_ep_150_rew_8.168549.pth" # DONE
#   "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-DIST-01:last_AnymalTerrain_ep_4800_rew_20.043377.pth" # DONE
#   "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-TERR-01:last_AnymalTerrain_ep_1800_rew_18.174595.pth" # DONE
#   "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-DISTTERR-01:last_AnymalTerrain_ep_4800_rew_14.132425.pth" # DONE
#   "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-0.5MASS-LSTM16-TERR-01:last_AnymalTerrain_ep_3200_rew_21.073418.pth" # DONE
# #   "A1Terrain_PPO_LSTM_NeuroRL:A1Terrain_NeuroRL_exp:A1-1.0MASS-LSTM16-TERR-01:last_A1Terrain_ep_4600_rew_16.256865.pth"

#     "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-FRONTIERSDISTTERR:last_AnymalTerrain_ep_3200_rew_20.145746.pth" # DONE
#     "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-FRONTIERSTERR:last_AnymalTerrain_ep_2900_rew_20.2482.pth" # DONE
#     "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-FRONTIERSDIST:last_AnymalTerrain_ep_900_rew_20.139568.pth" # DONE
#     "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-FRONTIERSBASELINE:last_AnymalTerrain_ep_700_rew_20.361492.pth" # DONE

#   "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-FRONTIERSDISTTERR:last_AnymalTerrain_ep_1100_rew_14.392729.pth" # DONE
#   "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-FRONTIERSDISTTERR:last_AnymalTerrain_ep_2200_rew_19.53241.pth" # DONE
#   "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-FRONTIERSDISTTERR:last_AnymalTerrain_ep_3800_rew_20.310041.pth" # DONE
#   "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-FRONTIERSDISTTERR:last_AnymalTerrain_ep_3900_rew_20.14785.pth" # DONE
#   "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-FRONTIERSDISTTERR:last_AnymalTerrain_ep_4000_rew_20.387749.pth" # DONE
#   "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-FRONTIERSDISTTERR:last_AnymalTerrain_ep_4100_rew_20.68903.pth" # DONE

  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-CORLDISTTERR:last_AnymalTerrain_ep_200_rew_6.1486754.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-CORLDISTTERR:last_AnymalTerrain_ep_300_rew_8.433804.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-CORLDISTTERR:last_AnymalTerrain_ep_400_rew_10.192444.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-CORLDISTTERR:last_AnymalTerrain_ep_500_rew_11.477056.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-CORLDISTTERR:last_AnymalTerrain_ep_600_rew_12.894477.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-CORLDISTTERR:last_AnymalTerrain_ep_700_rew_13.613478.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-CORLDISTTERR:last_AnymalTerrain_ep_800_rew_15.344866.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-CORLDISTTERR:last_AnymalTerrain_ep_900_rew_12.484828.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-CORLDISTTERR:last_AnymalTerrain_ep_1000_rew_15.300709.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-CORLDISTTERR:last_AnymalTerrain_ep_1100_rew_10.703082.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-CORLDISTTERR:last_AnymalTerrain_ep_1200_rew_13.709155.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-CORLDISTTERR:last_AnymalTerrain_ep_1300_rew_13.720617.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-CORLDISTTERR:last_AnymalTerrain_ep_1400_rew_15.347135.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-CORLDISTTERR:last_AnymalTerrain_ep_1500_rew_15.248126.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-CORLDISTTERR:last_AnymalTerrain_ep_1600_rew_14.338628.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-CORLDISTTERR:last_AnymalTerrain_ep_1700_rew_15.457918.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-CORLDISTTERR:last_AnymalTerrain_ep_1800_rew_16.685356.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-CORLDISTTERR:last_AnymalTerrain_ep_1900_rew_15.336959.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-CORLDISTTERR:last_AnymalTerrain_ep_2000_rew_16.601225.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-CORLDISTTERR:last_AnymalTerrain_ep_2500_rew_16.594769.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-CORLDISTTERR:last_AnymalTerrain_ep_3000_rew_14.874878.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-CORLDISTTERR:last_AnymalTerrain_ep_3500_rew_17.787632.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-CORLDISTTERR:last_AnymalTerrain_ep_3700_rew_20.14857.pth"

  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-CORLDISTTERR:last_AnymalTerrain_ep_200_rew_6.8250656.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-CORLDISTTERR:last_AnymalTerrain_ep_300_rew_10.119753.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-CORLDISTTERR:last_AnymalTerrain_ep_400_rew_12.110974.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-CORLDISTTERR:last_AnymalTerrain_ep_500_rew_12.495365.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-CORLDISTTERR:last_AnymalTerrain_ep_600_rew_13.912197.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-CORLDISTTERR:last_AnymalTerrain_ep_700_rew_14.312197.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-CORLDISTTERR:last_AnymalTerrain_ep_800_rew_15.107031.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-CORLDISTTERR:last_AnymalTerrain_ep_900_rew_13.206897.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-CORLDISTTERR:last_AnymalTerrain_ep_1000_rew_14.850766.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-CORLDISTTERR:last_AnymalTerrain_ep_1100_rew_11.663524.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-CORLDISTTERR:last_AnymalTerrain_ep_1200_rew_13.862656.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-CORLDISTTERR:last_AnymalTerrain_ep_1300_rew_15.301971.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-CORLDISTTERR:last_AnymalTerrain_ep_1400_rew_15.220953.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-CORLDISTTERR:last_AnymalTerrain_ep_1600_rew_15.04414.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-CORLDISTTERR:last_AnymalTerrain_ep_1700_rew_14.368644.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-CORLDISTTERR:last_AnymalTerrain_ep_1800_rew_16.166517.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-CORLDISTTERR:last_AnymalTerrain_ep_1900_rew_15.772427.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-CORLDISTTERR:last_AnymalTerrain_ep_2000_rew_16.889687.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-CORLDISTTERR:last_AnymalTerrain_ep_2100_rew_17.150562.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-CORLDISTTERR:last_AnymalTerrain_ep_2200_rew_16.620295.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-CORLDISTTERR:last_AnymalTerrain_ep_2300_rew_16.302753.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-CORLDISTTERR:last_AnymalTerrain_ep_2400_rew_15.608815.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-CORLDISTTERR:last_AnymalTerrain_ep_2500_rew_16.140593.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-CORLDISTTERR:last_AnymalTerrain_ep_2600_rew_18.350157.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-CORLDISTTERR:last_AnymalTerrain_ep_2700_rew_18.907959.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-CORLDISTTERR:last_AnymalTerrain_ep_2800_rew_17.199806.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-CORLDISTTERR:last_AnymalTerrain_ep_2900_rew_17.177887.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-CORLDISTTERR:last_AnymalTerrain_ep_3000_rew_16.622911.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-CORLDISTTERR:last_AnymalTerrain_ep_3100_rew_18.139688.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-CORLDISTTERR:last_AnymalTerrain_ep_3200_rew_18.644165.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-CORLDISTTERR:last_AnymalTerrain_ep_3300_rew_17.097162.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-CORLDISTTERR:last_AnymalTerrain_ep_3400_rew_19.06523.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-CORLDISTTERR:last_AnymalTerrain_ep_3500_rew_18.5328.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-CORLDISTTERR:last_AnymalTerrain_ep_3600_rew_16.928177.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-CORLDISTTERR:last_AnymalTerrain_ep_3700_rew_19.987637.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-CORLDISTTERR:last_AnymalTerrain_ep_3800_rew_18.467999.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-CORLDISTTERR:last_AnymalTerrain_ep_3900_rew_18.793278.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-CORLDISTTERR:last_AnymalTerrain_ep_4000_rew_18.484495.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-CORLDISTTERR:last_AnymalTerrain_ep_4100_rew_18.95446.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-CORLDISTTERR:last_AnymalTerrain_ep_4200_rew_19.041822.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-CORLDISTTERR:last_AnymalTerrain_ep_4300_rew_17.954895.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-CORLDISTTERR:last_AnymalTerrain_ep_4400_rew_18.998327.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-CORLDISTTERR:last_AnymalTerrain_ep_4500_rew_19.806221.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-CORLDISTTERR:last_AnymalTerrain_ep_4600_rew_18.554163.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-CORLDISTTERR:last_AnymalTerrain_ep_4700_rew_18.879852.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-CORLDISTTERR:last_AnymalTerrain_ep_4800_rew_19.64705.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-CORLDISTTERR:last_AnymalTerrain_ep_4900_rew_17.769163.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-CORLDISTTERR:last_AnymalTerrain_ep_5000_rew_16.690823.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-CORLDISTTERR:last_AnymalTerrain_ep_6000_rew_20.090017.pth"
  "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-CORLDISTTERR:last_AnymalTerrain_ep_6700_rew_20.21499.pth"

#   "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-CORLTERR:last_AnymalTerrain_ep_200_rew_6.420168.pth"
#   "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-CORLTERR:last_AnymalTerrain_ep_300_rew_8.896029.pth"
#   "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-CORLTERR:last_AnymalTerrain_ep_400_rew_10.528543.pth"
#   "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-CORLTERR:last_AnymalTerrain_ep_500_rew_13.228901.pth"
#   "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-CORLTERR:last_AnymalTerrain_ep_1000_rew_14.604733.pth"
#   "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-CORLTERR:last_AnymalTerrain_ep_1500_rew_14.298144.pth"
#   "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-CORLTERR:last_AnymalTerrain_ep_2000_rew_18.007153.pth"
#   "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-CORLTERR:last_AnymalTerrain_ep_2500_rew_18.825102.pth"
#   "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-CORLTERR:last_AnymalTerrain_ep_3000_rew_19.434002.pth"
#   "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM4-CORLTERR:last_AnymalTerrain_ep_3300_rew_20.003773.pth" # DONE

#   "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-CORLTERR:last_AnymalTerrain_ep_200_rew_5.884394.pth"
#   "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-CORLTERR:last_AnymalTerrain_ep_300_rew_7.6767497.pth"
#   "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-CORLTERR:last_AnymalTerrain_ep_400_rew_10.565976.pth"
#   "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-CORLTERR:last_AnymalTerrain_ep_600_rew_12.610853.pth"
#   "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-CORLTERR:last_AnymalTerrain_ep_1000_rew_14.291509.pth"
#   "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-CORLTERR:last_AnymalTerrain_ep_1500_rew_14.035113.pth"
#   "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-CORLTERR:last_AnymalTerrain_ep_2000_rew_16.989128.pth"
#   "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-CORLTERR:last_AnymalTerrain_ep_2500_rew_17.63955.pth"
#   "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-CORLTERR:last_AnymalTerrain_ep_3000_rew_18.42784.pth"
#   "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-CORLTERR:last_AnymalTerrain_ep_3500_rew_18.885078.pth"
#   "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-CORLTERR:last_AnymalTerrain_ep_3800_rew_20.163399.pth" # DONE

# For thesis, generating gradients for all actuators, to see if I can figure out what the other 5 key cx neurons are driving???
    # "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-FRONTIERSDISTTERR:last_AnymalTerrain_ep_3200_rew_20.145746.pth" # DONE
    # "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-FRONTIERSTERR:last_AnymalTerrain_ep_2900_rew_20.2482.pth" # DONE
    # "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-FRONTIERSDIST:last_AnymalTerrain_ep_900_rew_20.139568.pth" # DONE
    # "AnymalTerrain_PPO_LSTM_NeuroRL:AnymalTerrain_NeuroRL_exp:ANYMAL-1.0MASS-LSTM16-FRONTIERSBASELINE:last_AnymalTerrain_ep_700_rew_20.361492.pth" # DONE

)

steps_after_stance_begins_values=0
length_s_values=(0.4 0.4 0.4 0.4 0.4 0.1 0.1 0.1 0.1 0.1 0.08 0.02 0.02 0.02 0.02 0.02)
forceY_values=(-0.333 -0.5 -0.667 -0.833 -1 -1 -1.5 -2 -2.5 -3 -3.5 -4 -6 -8 -10 -12)
# length_s_values=(0.08)
# forceY_values=(-3.5)

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
        +output_path=$export_path/$model_type/evaluate_robustness_throughout_training/$steps_after_stance_begins/$length_s/$forceY/$model_name
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


