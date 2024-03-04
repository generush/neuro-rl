#!/bin/bash

hardcoded_strings=(
"ANYMAL-1.0MASS-LSTM16-DISTTERR-01_U-0.4-1.0-7-50_UNPERTURBED"
"ANYMAL-1.0MASS-LSTM16-DISTTERR-01_U-0.4-1.0-14-25_UNPERTURBED"
# "ANYMAL-1.0MASS-LSTM16-DISTTERR-02_U-0.4-1.0-7-50_UNPERTURBED"
# "ANYMAL-1.0MASS-LSTM16-DISTTERR-02_U-0.4-1.0-14-25_UNPERTURBED"
# "ANYMAL-1.0MASS-LSTM16-DISTTERR-03_U-0.4-1.0-7-50_UNPERTURBED"
# "ANYMAL-1.0MASS-LSTM16-DISTTERR-03_U-0.4-1.0-14-25_UNPERTURBED"

# "ANYMAL-1.0MASS-LSTM16-TERR-01_U-0.4-1.0-7-50_UNPERTURBED"
# "ANYMAL-1.0MASS-LSTM16-TERR-01_U-0.4-1.0-14-25_UNPERTURBED"
# "ANYMAL-1.0MASS-LSTM16-TERR-02_U-0.4-1.0-7-50_UNPERTURBED"
# "ANYMAL-1.0MASS-LSTM16-TERR-02_U-0.4-1.0-14-25_UNPERTURBED"
# "ANYMAL-1.0MASS-LSTM16-TERR-03_U-0.4-1.0-7-50_UNPERTURBED"
# "ANYMAL-1.0MASS-LSTM16-TERR-03_U-0.4-1.0-14-25_UNPERTURBED"

# "ANYMAL-1.0MASS-LSTM16-DIST-01_U-0.4-1.0-7-50_UNPERTURBED"
# "ANYMAL-1.0MASS-LSTM16-DIST-01_U-0.4-1.0-14-25_UNPERTURBED"
# "ANYMAL-1.0MASS-LSTM16-DIST-02_U-0.4-1.0-7-50_UNPERTURBED"
# "ANYMAL-1.0MASS-LSTM16-DIST-02_U-0.4-1.0-14-25_UNPERTURBED"
# "ANYMAL-1.0MASS-LSTM16-DIST-03_U-0.4-1.0-7-50_UNPERTURBED"
# "ANYMAL-1.0MASS-LSTM16-DIST-03_U-0.4-1.0-14-25_UNPERTURBED"

# "ANYMAL-1.0MASS-LSTM16-BASELINE-01_U-0.4-1.0-7-50_UNPERTURBED"
# "ANYMAL-1.0MASS-LSTM16-BASELINE-01_U-0.4-1.0-14-25_UNPERTURBED"
# "ANYMAL-1.0MASS-LSTM16-BASELINE-02_U-0.4-1.0-7-50_UNPERTURBED"
# "ANYMAL-1.0MASS-LSTM16-BASELINE-02_U-0.4-1.0-14-25_UNPERTURBED"
# "ANYMAL-1.0MASS-LSTM16-BASELINE-03_U-0.4-1.0-7-50_UNPERTURBED"
# "ANYMAL-1.0MASS-LSTM16-BASELINE-03_U-0.4-1.0-14-25_UNPERTURBED"
)

for hardcoded_string in "${hardcoded_strings[@]}"; do
    (
        # Operations to be performed, similar to the previous examples
        # The commands here are executed in a subshell due to the surrounding parentheses

        # GET THE MODEL NAME
        #----------------------------------------------------------------
        # Save the old IFS value
        oldIFS="$IFS"

        # Set IFS to underscore for splitting the hardcoded string
        IFS='_'

        # Read the split parts into an array
        read -ra ADDR <<< "$hardcoded_string"

        # Restore the old IFS
        IFS="$oldIFS"

        # Create a string that includes the first part of the hardcoded string
        model_name=${ADDR[0]}

        #----------------------------------------------------------------

        # Optionally, you can remove a potential file extension from the hardcoded string
        hardcoded_string="${hardcoded_string%.*}"

        # Call the Python script with the current hardcoded string
        python ../../analysis_pipeline.py --config_path "../../cfg/analyze/analysis.yaml" --input_path "../data/raw/${hardcoded_string}" --output_path "../data/processed/${hardcoded_string}"
    )
done

echo "All scripts have been executed sequentially."