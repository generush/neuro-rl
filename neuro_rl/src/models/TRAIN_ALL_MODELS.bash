#!/bin/bash

# Run the first script
./ANYMAL-1.0MASS-LSTM16-DISTTERR.bash
./ANYMAL-1.0MASS-LSTM16-TERR.bash
./ANYMAL-1.0MASS-LSTM16-DIST.bash
./ANYMAL-1.0MASS-LSTM16-BASELINE.bash

./ANYMAL-1.0MASS-FF-DISTTERR.bash
./ANYMAL-1.0MASS-FF-BASELINE.bash

# ./A1-1.0MASS-FF-DISTTERR.bash
# ./A1-1.0MASS-LSTM16-DISTTERR.bash


echo "All scripts have been executed sequentially."