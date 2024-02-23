
# https://plotly.com/python/3d-scatter-plots/

import matplotlib
matplotlib.use('TkAgg')  # Replace 'TkAgg' with another backend if needed

# https://datascience.stackexchange.com/questions/55066/how-to-export-pca-to-use-in-another-program

import os
from pdfCropMargins import crop


    
def crop_pdfs_in_folder(folder_path):
    # Create the "cropped" folder if it doesn't exist
    cropped_folder = os.path.join(folder_path, "cropped")
    os.makedirs(cropped_folder, exist_ok=True)

    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path) and file_name.endswith(".pdf"):
            crop([file_path, '-o', folder_path + '/cropped/'])
        else:
            print(f"Skipping file: {file_name} (not a PDF)")
            

# crop white space out of pdfs

path = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-06-05_10-56-19_u[0.3,1.0,16]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[50]/TRAJ_XY_DENSITY_10/'
crop_pdfs_in_folder(path)
path = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-06-05_10-56-19_u[0.3,1.0,16]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[50]/TRAJ_XY_DENSITY_50/'
crop_pdfs_in_folder(path)
path = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-06-06_08-53-16_u[0.4,1.0,14]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[50]/TRAJ_XY_DENSITY_10/'
crop_pdfs_in_folder(path)
path = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-06-06_08-53-16_u[0.4,1.0,14]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[50]/TRAJ_XY_DENSITY_50'
crop_pdfs_in_folder(path)





print('hi')