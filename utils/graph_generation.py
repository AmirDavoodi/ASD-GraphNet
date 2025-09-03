import os
import pickle
import networkx as nx
import pandas as pd
from typing import List
from p_tqdm import p_map
from typing import Union

SUBJECT_GRAPH_PATH = os.path.join(os.path.dirname(__file__), '..', 'subject_graphs', 'subjects_graphs')

def make_nx_graph(subject_idx_and_subject_data_df: tuple[int, pd.DataFrame]) -> nx.Graph:
    subject_idx, subject_data_df = subject_idx_and_subject_data_df
    pear_corr = subject_data_df.corr()
    G = nx.from_pandas_adjacency(pear_corr)
    G.remove_edges_from(list(nx.selfloop_edges(G)))
    return G

def apply_edge_weight_threshold(G: nx.Graph, edge_weight_threshold: float) -> nx.Graph:
    filtered_edge = [
        (node1, node2)
        for node1, node2, edge_attribute in G.edges(data=True)
        if abs(edge_attribute["weight"]) <= edge_weight_threshold
    ]
    G.remove_edges_from(filtered_edge)
    return G

def create_subjects_graph(df_subjects: pd.DataFrame, limit_subjects: int = 10, atlases: List[str] = ['cc200', 'aal', 'dos160'], edge_weight_threshold: Union[float, None] = None) -> pd.DataFrame:
    print("Making subjects graphs from files...")

    if limit_subjects == -1:
        limit_subjects = len(df_subjects)

    all_subjects_graphs = {}
    for atlas in atlases:
        if not os.path.exists(f"{SUBJECT_GRAPH_PATH}_{atlas}.pkl"):
            subjects_graphs: List[nx.Graph] = p_map(make_nx_graph, enumerate(df_subjects[atlas][:limit_subjects]), num_cpus=8)

            if edge_weight_threshold is not None:
                subjects_graphs = [apply_edge_weight_threshold(G, edge_weight_threshold) for G in subjects_graphs]

            with open(f"{SUBJECT_GRAPH_PATH}_{atlas}.pkl", "wb") as f:
                pickle.dump(subjects_graphs, f)
        else:
            with open(f"{SUBJECT_GRAPH_PATH}_{atlas}.pkl", "rb") as f:
                subjects_graphs: List[nx.Graph] = pickle.load(f)
        
        all_subjects_graphs[atlas] = subjects_graphs

    df_subjects_with_graphs = df_subjects.iloc[:limit_subjects].copy()
    for atlas in atlases:
        df_subjects_with_graphs[f'nx_graph_{atlas}'] = all_subjects_graphs[atlas]

    print("Graphs are successfully created :)")
    return df_subjects_with_graphs