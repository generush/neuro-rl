#!/bin/bash

# "ANYMAL-1.0MASS-LSTM16-DISTTERR-01_U-0.4-1.0-7-50_UNPERTURBED/"
# "ANYMAL-1.0MASS-LSTM16-DISTTERR-01_U-0.4-1.0-14-25_UNPERTURBED/"
# "ANYMAL-1.0MASS-LSTM16-DISTTERR-02_U-0.4-1.0-7-50_UNPERTURBED/"
# "ANYMAL-1.0MASS-LSTM16-DISTTERR-02_U-0.4-1.0-14-25_UNPERTURBED/"
# "ANYMAL-1.0MASS-LSTM16-DISTTERR-03_U-0.4-1.0-7-50_UNPERTURBED/"
# "ANYMAL-1.0MASS-LSTM16-DISTTERR-03_U-0.4-1.0-14-25_UNPERTURBED/"
# "ANYMAL-1.0MASS-LSTM16-TERR-01_U-0.4-1.0-7-50_UNPERTURBED/"
# "ANYMAL-1.0MASS-LSTM16-TERR-01_U-0.4-1.0-14-25_UNPERTURBED/"
# "ANYMAL-1.0MASS-LSTM16-TERR-02_U-0.4-1.0-7-50_UNPERTURBED/"
# "ANYMAL-1.0MASS-LSTM16-TERR-02_U-0.4-1.0-14-25_UNPERTURBED/"
# "ANYMAL-1.0MASS-LSTM16-TERR-03_U-0.4-1.0-7-50_UNPERTURBED/"
# "ANYMAL-1.0MASS-LSTM16-TERR-03_U-0.4-1.0-14-25_UNPERTURBED/"
# "ANYMAL-1.0MASS-LSTM16-DIST-01_U-0.4-1.0-7-50_UNPERTURBED/"
# "ANYMAL-1.0MASS-LSTM16-DIST-01_U-0.4-1.0-14-25_UNPERTURBED/"
# "ANYMAL-1.0MASS-LSTM16-DIST-02_U-0.4-1.0-7-50_UNPERTURBED/"
# "ANYMAL-1.0MASS-LSTM16-DIST-02_U-0.4-1.0-14-25_UNPERTURBED/"
# "ANYMAL-1.0MASS-LSTM16-DIST-03_U-0.4-1.0-7-50_UNPERTURBED/"
# "ANYMAL-1.0MASS-LSTM16-DIST-03_U-0.4-1.0-14-25_UNPERTURBED/"
# "ANYMAL-1.0MASS-LSTM16-BASELINE-01_U-0.4-1.0-7-50_UNPERTURBED/"
# "ANYMAL-1.0MASS-LSTM16-BASELINE-01_U-0.4-1.0-14-25_UNPERTURBED/"
# "ANYMAL-1.0MASS-LSTM16-BASELINE-02_U-0.4-1.0-7-50_UNPERTURBED/"
# "ANYMAL-1.0MASS-LSTM16-BASELINE-02_U-0.4-1.0-14-25_UNPERTURBED/"
# "ANYMAL-1.0MASS-LSTM16-BASELINE-03_U-0.4-1.0-7-50_UNPERTURBED/"
# "ANYMAL-1.0MASS-LSTM16-BASELINE-03_U-0.4-1.0-14-25_UNPERTURBED/"


# "ANYMAL-1.0MASS-LSTM16-DISTTERR-99_U-0.4-1.0-14-25_UNPERTURBED/"
# "ANYMAL-1.0MASS-LSTM16-DISTTERR-99_U-0.4-1.0-7-50_UNPERTURBED/"
# "ANYMAL-1.0MASS-LSTM16-DISTTERR-99_U-1.0-1.0-1-400_PERTURBED/"
# "ANYMAL-1.0MASS-LSTM16-DISTTERR-99_U-1.0-1.0-1-1_PERTURBED/"



# Define an array with combined model_type/run_name pairs
model_run_pairs=(
    # "ANYMAL-1.0MASS-LSTM16-DISTTERR-00:u-0.4_1.0-7-v-0._0.-1-r-0._0.-1-n-50"
    # "ANYMAL-1.0MASS-LSTM16-DISTTERR-00:u-0.4_1.0-14-v-0._0.-1-r-0._0.-1-n-25"
    # "ANYMAL-1.0MASS-LSTM16-DISTTERR-00:u-1.0_1.0-1-v-0._0.-1-r-0._0.-1-n-1"
    # "ANYMAL-1.0MASS-LSTM16-DISTTERR-00:u-1.0_1.0-1-v-0._0.-1-r-0._0.-1-n-400"

    # "ANYMAL-1.0MASS-LSTM16-DISTTERR-01:u-0.4_1.0-7-v-0._0.-1-r-0._0.-1-n-50"
    # "ANYMAL-1.0MASS-LSTM16-DISTTERR-01:u-0.4_1.0-14-v-0._0.-1-r-0._0.-1-n-25"
    # "ANYMAL-1.0MASS-LSTM16-DISTTERR-01:u-1.0_1.0-1-v-0._0.-1-r-0._0.-1-n-1"
    # "ANYMAL-1.0MASS-LSTM16-DISTTERR-01:u-1.0_1.0-1-v-0._0.-1-r-0._0.-1-n-400"

    # "ANYMAL-1.0MASS-LSTM16-DISTTERR-02:u-0.4_1.0-7-v-0._0.-1-r-0._0.-1-n-50"
    # "ANYMAL-1.0MASS-LSTM16-DISTTERR-02:u-0.4_1.0-14-v-0._0.-1-r-0._0.-1-n-25"
    # "ANYMAL-1.0MASS-LSTM16-DISTTERR-02:u-1.0_1.0-1-v-0._0.-1-r-0._0.-1-n-1"
    # "ANYMAL-1.0MASS-LSTM16-DISTTERR-02:u-1.0_1.0-1-v-0._0.-1-r-0._0.-1-n-400"

    # "ANYMAL-1.0MASS-LSTM16-TERR-00:u-0.4_1.0-7-v-0._0.-1-r-0._0.-1-n-50"
    # "ANYMAL-1.0MASS-LSTM16-TERR-00:u-0.4_1.0-14-v-0._0.-1-r-0._0.-1-n-25"
    # "ANYMAL-1.0MASS-LSTM16-TERR-00:u-1.0_1.0-1-v-0._0.-1-r-0._0.-1-n-1"
    # "ANYMAL-1.0MASS-LSTM16-TERR-00:u-1.0_1.0-1-v-0._0.-1-r-0._0.-1-n-400"

    # "ANYMAL-1.0MASS-LSTM16-TERR-01:u-0.4_1.0-7-v-0._0.-1-r-0._0.-1-n-50"
    # "ANYMAL-1.0MASS-LSTM16-TERR-01:u-0.4_1.0-14-v-0._0.-1-r-0._0.-1-n-25"
    # "ANYMAL-1.0MASS-LSTM16-TERR-01:u-1.0_1.0-1-v-0._0.-1-r-0._0.-1-n-1"
    # "ANYMAL-1.0MASS-LSTM16-TERR-01:u-1.0_1.0-1-v-0._0.-1-r-0._0.-1-n-400"

    # "ANYMAL-1.0MASS-LSTM16-TERR-02:u-0.4_1.0-7-v-0._0.-1-r-0._0.-1-n-50"
    # "ANYMAL-1.0MASS-LSTM16-TERR-02:u-0.4_1.0-14-v-0._0.-1-r-0._0.-1-n-25"
    # "ANYMAL-1.0MASS-LSTM16-TERR-02:u-1.0_1.0-1-v-0._0.-1-r-0._0.-1-n-1"
    # "ANYMAL-1.0MASS-LSTM16-TERR-02:u-1.0_1.0-1-v-0._0.-1-r-0._0.-1-n-400"


    # "ANYMAL-1.0MASS-LSTM16-TERR-122:u-0.4_1.0-7-v-0._0.-1-r-0._0.-1-n-50"


    # "ANYMAL-1.0MASS-LSTM16-TERR-01:u-0.4_1.0-7-v-0._0.-1-r-0._0.-1-n-20"

    # "ANYMAL-1.0MASS-LSTM4-TERR-01:U-0.4-1.0-14-20_UNPERTURBED"
    # "ANYMAL-1.0MASS-LSTM16-DISTTERR-01:u_0.4_1.0_14_v_0._0._1_r_0._0._1_n_20"
    # "ANYMAL-1.0MASS-LSTM4-DISTTERR-01:u_0.4_1.0_14_v_0._0._1_r_0._0._1_n_20"
    # "ANYMAL-1.0MASS-LSTM4-DISTTERR-01:u_0.4_1.0_14_v_0._0._1_r_0._0._1_n_201"

    # "ANYMAL-1.0MASS-LSTM16-DISTTERR-01:u_0.4_1.0_14_v_0._0._1_r_0._0._1_n_20"


    # "ANYMAL-1.0MASS-LSTM16-DISTTERR-01:u_0.4_1_14_v_0_0_1_r_0_0_1_n_20"
    # "ANYMAL-1.0MASS-LSTM16-DISTTERR-01:u_0.4_1_7_v_0_0_1_r_0_0_1_n_40"
    # "ANYMAL-1.0MASS-LSTM16-DISTTERR-01:u_0_0_1_v_0.4_1_7_r_0_0_1_n_40"
    # "ANYMAL-1.0MASS-LSTM16-DISTTERR-01:u_0_0_1_v_0_0_1_r_-1_1_7_n_40"
    # "ANYMAL-1.0MASS-LSTM16-DISTTERR-01:u_0_0_1_v_0_0_1_r_1_1_1_n_1"
    # "ANYMAL-1.0MASS-LSTM16-DISTTERR-01:u_0_0_1_v_0_0_1_r_-1_-1_1_n_1"
    # "ANYMAL-1.0MASS-LSTM16-DISTTERR-01:u_0_0_1_v_-1_-1_1_r_0_0_1_n_1"
    # "ANYMAL-1.0MASS-LSTM16-DISTTERR-01:u_0_0_1_v_1_1_1_r_0_0_1_n_1"
    # "ANYMAL-1.0MASS-LSTM16-DISTTERR-01:u_-1_-1_1_v_0_0_1_r_0_0_1_n_1"
    # "ANYMAL-1.0MASS-LSTM16-DISTTERR-01:u_1_1_1_v_0_0_1_r_0_0_1_n_1"
    # "ANYMAL-1.0MASS-LSTM4-DISTTERR-01:u_0.4_1_14_v_0_0_1_r_0_0_1_n_20"

    # "ANYMAL-1.0MASS-LSTM4-DISTTERR-01:u_0.4_1_7_v_0_0_1_r_0_0_1_n_40"
    # "ANYMAL-1.0MASS-LSTM4-DISTTERR-01:u_0_0_1_v_0.4_1_7_r_0_0_1_n_40"
    # "ANYMAL-1.0MASS-LSTM4-DISTTERR-01:u_0_0_1_v_0_0_1_r_-1_1_7_n_40"
    # "ANYMAL-1.0MASS-LSTM4-DISTTERR-01:u_0_0_1_v_0_0_1_r_1_1_1_n_1"
    # "ANYMAL-1.0MASS-LSTM4-DISTTERR-01:u_0_0_1_v_0_0_1_r_-1_-1_1_n_1"
    # "ANYMAL-1.0MASS-LSTM4-DISTTERR-01:u_0_0_1_v_-1_-1_1_r_0_0_1_n_1"
    # "ANYMAL-1.0MASS-LSTM4-DISTTERR-01:u_0_0_1_v_1_1_1_r_0_0_1_n_1"
    # "ANYMAL-1.0MASS-LSTM4-DISTTERR-01:u_-1_-1_1_v_0_0_1_r_0_0_1_n_1"
    # "ANYMAL-1.0MASS-LSTM4-DISTTERR-01:u_1_1_1_v_0_0_1_r_0_0_1_n_1"

    # "A1-1.0MASS-LSTM16-TERR-228:u_0.4_1_14_v_0_0_1_r_0_0_1_n_20"
    # "A1-1.0MASS-LSTM16-TERR-228:u_0.4_1_28_v_0_0_1_r_0_0_1_n_10"
    # "A1-1.0MASS-LSTM16-TERR-01:u_0.4_1_28_v_0_0_1_r_0_0_1_n_10"
    # "ANYMAL-0.5MASS-LSTM16-TERR-01:u_0.4_1_28_v_0_0_1_r_0_0_1_n_10"
    # "ANYMAL-1.0MASS-LSTM16-DIST-01:u_0.4_1_28_v_0_0_1_r_0_0_1_n_10"
    # "2024-03-14-01-16_A1Terrain:u_0.4_1_28_v_0_0_1_r_0_0_1_n_10"
    # "ANYMAL-0.5MASS-LSTM16-TERR-01:u_0.4_1_28_v_0_0_1_r_0_0_1_n_10"

    # "ANYMAL-1.0MASS-LSTM16-BASELINE-01:last_AnymalTerrain_ep_1000_rew_20.962988.pth:u_0.4_1_28_v_0_0_1_r_0_0_1_n_15"
    # "ANYMAL-1.0MASS-LSTM16-DIST-01:model:last_AnymalTerrain_ep_5000_rew_16.480799.pth:u_0.4_1_28_v_0_0_1_r_0_0_1_n_15"
    # "ANYMAL-1.0MASS-LSTM16-TERR-01:last_AnymalTerrain_ep_2000_rew_18.73817.pth:u_0.4_1_28_v_0_0_1_r_0_0_1_n_15"
    # "ANYMAL-1.0MASS-LSTM16-DISTTERR-01:last_AnymalTerrain_ep_4600_rew_15.199695.pth:u_0.4_1_28_v_0_0_1_r_0_0_1_n_15"
    # "ANYMAL-1.0MASS-LSTM4-BASELINE-01:last_AnymalTerrain_ep_150_rew_8.168549.pth:u_0.4_1_28_v_0_0_1_r_0_0_1_n_15"
    # "ANYMAL-1.0MASS-LSTM4-DIST-01:last_AnymalTerrain_ep_4800_rew_20.043377.pth:u_0.4_1_28_v_0_0_1_r_0_0_1_n_15"
    # "ANYMAL-1.0MASS-LSTM4-TERR-01:last_AnymalTerrain_ep_1800_rew_18.174595.pth:u_0.4_1_28_v_0_0_1_r_0_0_1_n_15"
    # "ANYMAL-1.0MASS-LSTM4-DISTTERR-01:last_AnymalTerrain_ep_4800_rew_14.132425.pth:u_0.4_1_28_v_0_0_1_r_0_0_1_n_15"
    # "ANYMAL-0.5MASS-LSTM16-TERR-01:last_AnymalTerrain_ep_3200_rew_21.073418.pth:u_0.4_1_28_v_0_0_1_r_0_0_1_n_15"

    # "A1-1.0MASS-LSTM16-TERR-01:last_A1Terrain_ep_4600_rew_16.256865.pth:u_0.4_1_28_v_0_0_1_r_0_0_1_n_15"



    # "A1-1.0MASS-LSTM16-TERR-01:last_A1Terrain_ep_4600_rew_16.256865.pth:u_0.6_1_30_v_0_0_1_r_0_0_1_n_15"
    # "A1-1.0MASS-LSTM16-TERR-01:last_A1Terrain_ep_6550_rew_17.543756.pth:u_0.6_1_30_v_0_0_1_r_0_0_1_n_15"
    # "A1-1.0MASS-LSTM16-TERR-01:last_A1Terrain_ep_14000_rew_19.912346.pth:u_0.6_1_30_v_0_0_1_r_0_0_1_n_15"


    # "ANYMAL-1.0MASS-LSTM4-DISTTERR-01:last_AnymalTerrain_ep_1200_rew_12.890905.pth:u_0.4_1_28_v_0_0_1_r_0_0_1_n_15"

#   "ANYMAL-1.0MASS-LSTM16-FRONTIERSDISTTERR:last_AnymalTerrain_ep_1100_rew_14.392729.pth:u_0.4_1_28_v_0_0_1_r_0_0_1_n_15"
#   "ANYMAL-1.0MASS-LSTM16-FRONTIERSDISTTERR:last_AnymalTerrain_ep_2200_rew_19.53241.pth:u_0.4_1_28_v_0_0_1_r_0_0_1_n_15"
#   "ANYMAL-1.0MASS-LSTM16-FRONTIERSDISTTERR:last_AnymalTerrain_ep_3800_rew_20.310041.pth:u_0.4_1_28_v_0_0_1_r_0_0_1_n_15"
#   "ANYMAL-1.0MASS-LSTM16-FRONTIERSDISTTERR:last_AnymalTerrain_ep_3900_rew_20.14785.pth:u_0.4_1_28_v_0_0_1_r_0_0_1_n_15"
#   "ANYMAL-1.0MASS-LSTM16-FRONTIERSDISTTERR:last_AnymalTerrain_ep_4000_rew_20.387749.pth:u_0.4_1_28_v_0_0_1_r_0_0_1_n_15"
#   "ANYMAL-1.0MASS-LSTM16-FRONTIERSDISTTERR:last_AnymalTerrain_ep_4100_rew_20.68903.pth:u_0.4_1_28_v_0_0_1_r_0_0_1_n_15"

  "ANYMAL-1.0MASS-LSTM4-CORLDISTTERR:last_AnymalTerrain_ep_200_rew_6.8250656.pth:u_0.4_1_28_v_0_0_1_r_0_0_1_n_15"
  "ANYMAL-1.0MASS-LSTM4-CORLDISTTERR:last_AnymalTerrain_ep_300_rew_10.119753.pth:u_0.4_1_28_v_0_0_1_r_0_0_1_n_15"
  "ANYMAL-1.0MASS-LSTM4-CORLDISTTERR:last_AnymalTerrain_ep_400_rew_12.110974.pth:u_0.4_1_28_v_0_0_1_r_0_0_1_n_15"
  "ANYMAL-1.0MASS-LSTM4-CORLDISTTERR:last_AnymalTerrain_ep_500_rew_12.495365.pth:u_0.4_1_28_v_0_0_1_r_0_0_1_n_15"
  "ANYMAL-1.0MASS-LSTM4-CORLDISTTERR:last_AnymalTerrain_ep_1000_rew_14.850766.pth:u_0.4_1_28_v_0_0_1_r_0_0_1_n_15"
  "ANYMAL-1.0MASS-LSTM4-CORLDISTTERR:last_AnymalTerrain_ep_2000_rew_16.889687.pth:u_0.4_1_28_v_0_0_1_r_0_0_1_n_15"
  "ANYMAL-1.0MASS-LSTM4-CORLDISTTERR:last_AnymalTerrain_ep_3000_rew_16.622911.pth:u_0.4_1_28_v_0_0_1_r_0_0_1_n_15"
  "ANYMAL-1.0MASS-LSTM4-CORLDISTTERR:last_AnymalTerrain_ep_4000_rew_18.484495.pth:u_0.4_1_28_v_0_0_1_r_0_0_1_n_15"
  "ANYMAL-1.0MASS-LSTM4-CORLDISTTERR:last_AnymalTerrain_ep_5000_rew_16.690823.pth:u_0.4_1_28_v_0_0_1_r_0_0_1_n_15"
  "ANYMAL-1.0MASS-LSTM4-CORLDISTTERR:last_AnymalTerrain_ep_6000_rew_20.090017.pth:u_0.4_1_28_v_0_0_1_r_0_0_1_n_15"
  "ANYMAL-1.0MASS-LSTM4-CORLDISTTERR:last_AnymalTerrain_ep_6700_rew_20.21499.pth:u_0.4_1_28_v_0_0_1_r_0_0_1_n_15"
  "ANYMAL-1.0MASS-LSTM16-CORLDISTTERR:last_AnymalTerrain_ep_200_rew_6.1486754.pth:u_0.4_1_28_v_0_0_1_r_0_0_1_n_15"
  "ANYMAL-1.0MASS-LSTM16-CORLDISTTERR:last_AnymalTerrain_ep_300_rew_8.433804.pth:u_0.4_1_28_v_0_0_1_r_0_0_1_n_15"
  "ANYMAL-1.0MASS-LSTM16-CORLDISTTERR:last_AnymalTerrain_ep_400_rew_10.192444.pth:u_0.4_1_28_v_0_0_1_r_0_0_1_n_15"
  "ANYMAL-1.0MASS-LSTM16-CORLDISTTERR:last_AnymalTerrain_ep_500_rew_11.477056.pth:u_0.4_1_28_v_0_0_1_r_0_0_1_n_15"
  "ANYMAL-1.0MASS-LSTM16-CORLDISTTERR:last_AnymalTerrain_ep_1000_rew_15.300709.pth:u_0.4_1_28_v_0_0_1_r_0_0_1_n_15"
  "ANYMAL-1.0MASS-LSTM16-CORLDISTTERR:last_AnymalTerrain_ep_1500_rew_15.248126.pth:u_0.4_1_28_v_0_0_1_r_0_0_1_n_15"
  "ANYMAL-1.0MASS-LSTM16-CORLDISTTERR:last_AnymalTerrain_ep_2000_rew_16.601225.pth:u_0.4_1_28_v_0_0_1_r_0_0_1_n_15"
  "ANYMAL-1.0MASS-LSTM16-CORLDISTTERR:last_AnymalTerrain_ep_2500_rew_16.594769.pth:u_0.4_1_28_v_0_0_1_r_0_0_1_n_15"
  "ANYMAL-1.0MASS-LSTM16-CORLDISTTERR:last_AnymalTerrain_ep_3000_rew_14.874878.pth:u_0.4_1_28_v_0_0_1_r_0_0_1_n_15"
  "ANYMAL-1.0MASS-LSTM16-CORLDISTTERR:last_AnymalTerrain_ep_3500_rew_17.787632.pth:u_0.4_1_28_v_0_0_1_r_0_0_1_n_15"
  "ANYMAL-1.0MASS-LSTM16-CORLDISTTERR:last_AnymalTerrain_ep_3700_rew_20.14857.pth:u_0.4_1_28_v_0_0_1_r_0_0_1_n_15"
  "ANYMAL-1.0MASS-LSTM4-CORLTERR:last_AnymalTerrain_ep_200_rew_6.420168.pth:u_0.4_1_28_v_0_0_1_r_0_0_1_n_15"
  "ANYMAL-1.0MASS-LSTM4-CORLTERR:last_AnymalTerrain_ep_300_rew_8.896029.pth:u_0.4_1_28_v_0_0_1_r_0_0_1_n_15"
  "ANYMAL-1.0MASS-LSTM4-CORLTERR:last_AnymalTerrain_ep_400_rew_10.528543.pth:u_0.4_1_28_v_0_0_1_r_0_0_1_n_15"
  "ANYMAL-1.0MASS-LSTM4-CORLTERR:last_AnymalTerrain_ep_500_rew_13.228901.pth:u_0.4_1_28_v_0_0_1_r_0_0_1_n_15"
  "ANYMAL-1.0MASS-LSTM4-CORLTERR:last_AnymalTerrain_ep_1000_rew_14.604733.pth:u_0.4_1_28_v_0_0_1_r_0_0_1_n_15"
  "ANYMAL-1.0MASS-LSTM4-CORLTERR:last_AnymalTerrain_ep_1500_rew_14.298144.pth:u_0.4_1_28_v_0_0_1_r_0_0_1_n_15"
  "ANYMAL-1.0MASS-LSTM4-CORLTERR:last_AnymalTerrain_ep_2000_rew_18.007153.pth:u_0.4_1_28_v_0_0_1_r_0_0_1_n_15"
  "ANYMAL-1.0MASS-LSTM4-CORLTERR:last_AnymalTerrain_ep_2500_rew_18.825102.pth:u_0.4_1_28_v_0_0_1_r_0_0_1_n_15"
  "ANYMAL-1.0MASS-LSTM4-CORLTERR:last_AnymalTerrain_ep_3000_rew_19.434002.pth:u_0.4_1_28_v_0_0_1_r_0_0_1_n_15"
  "ANYMAL-1.0MASS-LSTM4-CORLTERR:last_AnymalTerrain_ep_3300_rew_20.003773.pth:u_0.4_1_28_v_0_0_1_r_0_0_1_n_15"
  "ANYMAL-1.0MASS-LSTM16-CORLTERR:last_AnymalTerrain_ep_200_rew_5.884394.pth:u_0.4_1_28_v_0_0_1_r_0_0_1_n_15"
  "ANYMAL-1.0MASS-LSTM16-CORLTERR:last_AnymalTerrain_ep_300_rew_7.6767497.pth:u_0.4_1_28_v_0_0_1_r_0_0_1_n_15"
  "ANYMAL-1.0MASS-LSTM16-CORLTERR:last_AnymalTerrain_ep_400_rew_10.565976.pth:u_0.4_1_28_v_0_0_1_r_0_0_1_n_15"
  "ANYMAL-1.0MASS-LSTM16-CORLTERR:last_AnymalTerrain_ep_600_rew_12.610853.pth:u_0.4_1_28_v_0_0_1_r_0_0_1_n_15"
  "ANYMAL-1.0MASS-LSTM16-CORLTERR:last_AnymalTerrain_ep_1000_rew_14.291509.pth:u_0.4_1_28_v_0_0_1_r_0_0_1_n_15"
  "ANYMAL-1.0MASS-LSTM16-CORLTERR:last_AnymalTerrain_ep_1500_rew_14.035113.pth:u_0.4_1_28_v_0_0_1_r_0_0_1_n_15"
  "ANYMAL-1.0MASS-LSTM16-CORLTERR:last_AnymalTerrain_ep_2000_rew_16.989128.pth:u_0.4_1_28_v_0_0_1_r_0_0_1_n_15"
  "ANYMAL-1.0MASS-LSTM16-CORLTERR:last_AnymalTerrain_ep_2500_rew_17.63955.pth:u_0.4_1_28_v_0_0_1_r_0_0_1_n_15"
  "ANYMAL-1.0MASS-LSTM16-CORLTERR:last_AnymalTerrain_ep_3000_rew_18.42784.pth:u_0.4_1_28_v_0_0_1_r_0_0_1_n_15"
  "ANYMAL-1.0MASS-LSTM16-CORLTERR:last_AnymalTerrain_ep_3500_rew_18.885078.pth:u_0.4_1_28_v_0_0_1_r_0_0_1_n_15"
  "ANYMAL-1.0MASS-LSTM16-CORLTERR:last_AnymalTerrain_ep_3800_rew_20.163399.pth:u_0.4_1_28_v_0_0_1_r_0_0_1_n_15"

)

# Loop through the model/run pairs
for pair in "${model_run_pairs[@]}"; do
    # Split the model and run using the '/' delimiter
    IFS=':' read -r model_type model_name run_name <<< "$pair"

    echo "Processing Model Type: $model_type / Model Name: $model_name / Run: $run_name"

    # Call the Python script with the current model and run
    python ../../analysis_pipeline_cycle_avg.py --config_path "../../cfg/analyze/analysis.yaml" --model_path "../../models/$model_type/nn/$model_name" --data_path "../../data/raw/$model_type/$run_name/$model_name/" --output_path "../../data/processed/$model_type/$run_name/$model_name"
done

echo "All scripts have been executed sequentially."