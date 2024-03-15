import os
import pandas as pd

def aggregate_data(root_folder):
    data = {}

    for subdir, dirs, files in os.walk(root_folder):
        found_csv = False  # Flag to track if valid CSV files are found in the directory

        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(subdir, file)

                try:
                    df = pd.read_csv(file_path, header=0)
                    value = df['0'].iloc[0]
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")
                    continue

                # Check the file type and update the corresponding value
                if 'recoveries' in file_path.lower() or 'trials' in file_path.lower():
                    found_csv = True  # Valid CSV found, set the flag
                    # Initialize directory entry if it doesn't exist
                    if subdir not in data:
                        data[subdir] = {'Recoveries': pd.NA, 'Trials': pd.NA}

                    if 'recoveries' in file_path.lower():
                        data[subdir]['Recoveries'] = value
                    elif 'trials' in file_path.lower():
                        data[subdir]['Trials'] = value

        # If no valid CSV files were found in the directory, don't add an entry
        if not found_csv and subdir in data:
            del data[subdir]  # Remove the directory entry if no CSV files were found

    # Convert the aggregated data to a DataFrame
    main_df = pd.DataFrame.from_dict(data, orient='index', columns=['Recoveries', 'Trials']).reset_index()
    main_df.rename(columns={'index': 'FolderPath'}, inplace=True)

    return main_df

# Update with your root folder path
root_folder = '../../neuro-rl/neuro_rl/data/raw'  # Update this path to your root folder
main_df = aggregate_data(root_folder)

# Optional: Display the DataFrame to verify the contents
print(main_df)

# Optional: Save to CSV
main_df.to_csv('aggregated_data.csv', index=False)