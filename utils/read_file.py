import os
import pandas as pd
import numpy as np
import nilearn.plotting as nip

def get_subjects_df(data_directory: str = "atlases", labels_file_path: str = "pheno_file.csv"):
    atlases = ['cc200', 'aal', 'dos160']

    # Create an empty dictionary to hold data for each subject
    subject_dict = {}

    for atlas in atlases:
        for subject_file in os.listdir(os.path.join(data_directory, atlas)):
            subject_path = os.path.join(data_directory, atlas, subject_file)
            subject_data = pd.read_csv(subject_path, sep="\t")
            subject_id = subject_file.split("_")[-3][2:]

            # If subject is not already in the dictionary, add it
            if subject_id not in subject_dict:
                subject_dict[subject_id] = {'id': subject_id}

            # Add data for the current atlas to the dictionary
            subject_dict[subject_id][atlas] = subject_data
            
    # Convert the dictionary to a list of dictionaries
    subject_list = list(subject_dict.values())

    df_subjects = pd.DataFrame(subject_list)

    df_labels = pd.read_csv(labels_file_path)
    df_labels.DX_GROUP = df_labels.DX_GROUP.map({1: 1, 2: 0})
    df_labels.SITE_ID
    df_labels['subject'] = df_labels['subject'].astype(str)

    df_subjects = df_subjects.merge(df_labels[['subject', 'DX_GROUP', 'SITE_ID']], left_on='id', right_on='subject', how='left')
    df_subjects = df_subjects.drop(columns=['subject'])
    df_subjects = df_subjects.rename(columns={'DX_GROUP': 'ASD'})

    return df_subjects

def extract_center_of_mass(csv_file_path: str = "CC200_ROI_labels.csv") -> np.ndarray:
    """
    Extract the center of mass from the CC200 ROI labels CSV file.

    Args:
    csv_file_path (str): Path to the CC200 ROI labels CSV file.

    Returns:
    np.ndarray: Center of mass coordinates (3D array).
    """
    # Read the CSV file using pandas
    ROI_labels_df = pd.read_csv(csv_file_path)

    # Extract the "center of mass" column and convert it to a NumPy array
    center_of_mass = ROI_labels_df[" center of mass"].str.strip(" ()").str.split(";", expand=True).astype(float)

    return center_of_mass.values

def get_aal_atlas_coordinate(path_to_aal_nii_file: str = "./aal_roi_atlas.nii.gz"):
    aal_coordinates = nip.find_parcellation_cut_coords(labels_img="./aal_roi_atlas.nii.gz")
    return aal_coordinates

def get_dos160_atlas_coordinate(path_to_dos160_nii_file: str = "./dos160_roi_atlas.nii.gz"):
    dos160_coordinates = nip.find_parcellation_cut_coords(labels_img="./dos160_roi_atlas.nii.gz")
    return dos160_coordinates