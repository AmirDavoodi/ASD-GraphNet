import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx
import nilearn.plotting as nip
import plotly.io as pio
from typing import Any, Union
pio.renderers.default = 'notebook'


def plot_site_id_distribution(df_subjects, output_dir='outputs'):
    """
    Plot the distribution of subjects for each site and save the plot to a file.

    Parameters:
    df_subjects (pd.DataFrame): DataFrame containing subject data
    output_dir (str): Directory to save the plot file (default: 'outputs')

    Returns:
    None
    """
    # Find unique values in SITE_ID column and their counts
    SITE_ID_unique_values_counts = df_subjects.SITE_ID.value_counts()
    SITE_ID_unique_values_counts_df = SITE_ID_unique_values_counts.reset_index()
    SITE_ID_unique_values_counts_df.columns = ['SITE_ID', 'Subject Counts']

    # Plotting the histogram
    plt.figure(figsize=(10, 6))
    plt.bar(SITE_ID_unique_values_counts_df['SITE_ID'], SITE_ID_unique_values_counts_df['Subject Counts'], color='skyblue')
    plt.xlabel('Site ID')
    plt.ylabel('Number of Subjects')
    plt.title('Distribution of Subjects for Each Site')
    plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
    plt.tight_layout()

    # Save the plot to a file
    output_file = os.path.join(output_dir,'site_id_distribution.png')
    plt.savefig(output_file)
    plt.close()  # Close the plot to free up memory

def plot_adjacency_matrix(df, patient_index, atlas_name):
    """
    Plots the adjacency matrix of a networkx graph for the patient at the specified index in a dataframe.

    Args:
        df: The pandas dataframe containing the data.
        patient_index: The index of the patient (ith row).
        atlas_name: The name of the atlas (cc200, aal, or dos160).
    """
    # Get the networkx graph object
    graph = df.loc[patient_index, f"nx_graph_{atlas_name}"]

    # Convert the graph to an adjacency matrix
    A = nx.adjacency_matrix(graph).todense()

    # Get patient information
    patient_id = df.loc[patient_index, "id"]
    site_id = df.loc[patient_index, "SITE_ID"]
    asd = df.loc[patient_index, "ASD"]

    # Create the figure with desired size (modify width and height as needed)
    fig, ax = plt.subplots(figsize=(5, 5))  # Increase figsize for larger image

    # Create the heatmap
    im = ax.matshow(A, cmap="RdBu_r")
    plt.colorbar(im, label="Edge Weight", fraction=0.03)
    plt.title(f"Correlation Matrix\n{atlas_name.upper()}\nPatient ID: {patient_id}, SITE_ID: {site_id}, ASD: {asd}")
    plt.xlabel("Nodes")
    plt.ylabel("Nodes")
    # Reduce label frequency (modify n as needed)
    n = 20  # Show labels every 20th node
    plt.xticks(range(0, len(A), n), [str(i) for i in range(0, len(A), n)], rotation=45, fontsize=6)
    plt.yticks(range(0, len(A), n), [str(i) for i in range(0, len(A), n)], fontsize=6)
    # plt.xticks([str(i) for i in ranpatient_index=1ge(0, len(A), n)], rotation=45, fontsize=6)
    # plt.yticks([str(i) for i in range(0, len(A), n)], fontsize=6)
    plt.tight_layout()

    # Save the plot
    output_path = os.path.join(
        os.path.dirname(__file__), 
        '..', 
        'outputs', 
        f"{patient_id}_{atlas_name}_adjacency_matrix.png")
    plt.savefig(output_path)

    # Close the plot to free up memory
    plt.close()

def plot_brain_connectome(df: pd.DataFrame, center_of_mass: np.ndarray, patient_index: int = 1, column_name: str = "nx_graph_cc200"):
    """
    Plot the brain connectome graph of a patient.

    Args:
    patient_index (int): Index of the patient in the dataframe.
    df (pd.DataFrame): Dataframe containing patient data.
    column_name (str): Column name in the dataframe containing the networkx graph.
    center_of_mass (np.ndarray): 3D array of center of mass coordinates for each ROI.

    Returns:
    None
    """
    row = df.iloc[patient_index]

    # Convert the networkx graph to a connectivity matrix
    connectivity_matrix = nx.to_numpy_array(row[column_name])

    # Plot the connectome on the brain image
    output_file = os.path.join(
        os.path.dirname(__file__), 
        '..', 
        'outputs', 
        f"connectome_patient_{row['id']}_{column_name}.png")
    nip.plot_connectome(connectivity_matrix, node_coords=center_of_mass, 
                         edge_threshold=0.1,
                         display_mode='ortho',
                         title=f"{column_name.upper()} Connectome - Patient {row['id']} (SITE_ID: {row['SITE_ID']}, ASD: {row['ASD']})",
                         output_file=output_file)

    # Display the plot
    plt.show()

def plot_3d_connectome(df: pd.DataFrame, center_of_mass: Union[tuple[np.ndarray, list[Any]], np.ndarray], patient_index: int = 1, column_name: str = "nx_graph_cc200", open_in_web = False):
    """
    Plot the 3D connectome graph of a patient.

    Args:
    patient_index (int): Index of the patient in the dataframe.
    df (pd.DataFrame): Dataframe containing patient data.
    column_name (str): Column name in the dataframe containing the connectivity matrix.
    center_of_mass (np.ndarray): 3D array of center of mass coordinates for each ROI.

    Returns:
    None
    """
    row = df.iloc[patient_index]

    # Convert the networkx graph to a connectivity matrix
    connectivity_matrix = nx.to_numpy_array(row[column_name])

    # Plot the 3D connectome
    view = nip.view_connectome(connectivity_matrix, node_coords=center_of_mass, 
                         edge_threshold=0.49,
                         node_color='black',
                         title=f"{column_name.upper()} Connectome - Patient {row['id']} (SITE_ID: {row['SITE_ID']}, ASD: {row['ASD']})",
                         title_fontsize=12)
    if open_in_web:
        view.open_in_browser()
        return
    return view