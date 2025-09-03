from typing import List, Union, Mapping, Any
import importlib
import pandas as pd
import networkx as nx
from networkx.classes.reportviews import DegreeView
from utils import graph_generation
from tqdm import tqdm
importlib.reload(graph_generation)

def nodes_degree(nx_graph: nx.Graph, edge_weight_threshold: Union[float, None] = None) -> List[int]:
    filtered_graph = nx_graph
    if edge_weight_threshold is not None:
        filtered_graph = graph_generation.apply_edge_weight_threshold(nx_graph, edge_weight_threshold)
    degree_view = filtered_graph.degree
    if isinstance(degree_view, DegreeView):
        return [deg for _, deg in degree_view]
    return []

def node_degree_centrality(nx_graph: nx.Graph, edge_weight_threshold: Union[float, None] = None) -> List[float]:
    filtered_graph = nx_graph
    if edge_weight_threshold is not None:
        filtered_graph = graph_generation.apply_edge_weight_threshold(nx_graph, edge_weight_threshold)
    degree_centrality = nx.degree_centrality(filtered_graph)
    return list(degree_centrality.values())

def node_eigenvector_centrality(nx_graph: nx.Graph, edge_weight_threshold: Union[float, None] = None) -> List[float]:
    filtered_graph = nx_graph
    if edge_weight_threshold is not None:
        filtered_graph = graph_generation.apply_edge_weight_threshold(nx_graph, edge_weight_threshold)
    eigenvector_centrality = nx.eigenvector_centrality(filtered_graph, max_iter=10000)
    return list(eigenvector_centrality.values())

def node_closeness_centrality(nx_graph: nx.Graph, edge_weight_threshold: Union[float, None] = None) -> List[float]:
    filtered_graph = nx_graph
    if edge_weight_threshold is not None:
        filtered_graph = graph_generation.apply_edge_weight_threshold(nx_graph, edge_weight_threshold)
    closeness_centrality = nx.closeness_centrality(filtered_graph)
    return list(closeness_centrality.values())

def node_betweenness_centrality(nx_graph: nx.Graph, edge_weight_threshold: Union[float, None] = None) -> List[float]:
    filtered_graph = nx_graph
    if edge_weight_threshold is not None:
        filtered_graph = graph_generation.apply_edge_weight_threshold(nx_graph, edge_weight_threshold)
    betweenness_centrality = nx.betweenness_centrality(filtered_graph)
    return list(betweenness_centrality.values())

# Define a function to compute node-based features
def node_features(nx_graph: nx.Graph, edge_weight_threshold: Union[float, None] = None) -> dict[str, Union[List[float], List[int]]]:
    filtered_graph = nx_graph
    if edge_weight_threshold is not None:
        filtered_graph = graph_generation.apply_edge_weight_threshold(nx_graph, edge_weight_threshold)
    return {
    'degree': nodes_degree(filtered_graph),
    'degree_centrality': node_degree_centrality(filtered_graph),
    'eigenvector_centrality': node_eigenvector_centrality(filtered_graph),
    'closeness_centrality': node_closeness_centrality(filtered_graph),
    'betweenness_centrality': node_betweenness_centrality(filtered_graph),
}

def compute_node_based_features(df_subjects: pd.DataFrame, edge_weight_threshold: float) -> pd.DataFrame:
    # Enable tqdm integration with Pandas
    tqdm.pandas()

    # Apply the function to each graph and store the results in a new dataframe
    node_based_features = df_subjects[['id', 'ASD', 'SITE_ID']].copy()

    for atlas in tqdm(['cc200', 'aal', 'dos160'], desc='Computing node-based features'):
        node_features_for_atlas = df_subjects[f'nx_graph_{atlas}'].progress_apply(node_features, args=(edge_weight_threshold,))
        node_based_features[f'{atlas}_degree'] = node_features_for_atlas.progress_apply(lambda x: x['degree'])
        node_based_features[f'{atlas}_degree_centrality'] = node_features_for_atlas.progress_apply(lambda x: x['degree_centrality'])
        node_based_features[f'{atlas}_eigenvector_centrality'] = node_features_for_atlas.progress_apply(lambda x: x['eigenvector_centrality'])
        node_based_features[f'{atlas}_closeness_centrality'] = node_features_for_atlas.progress_apply(lambda x: x['closeness_centrality'])
        node_based_features[f'{atlas}_betweenness_centrality'] = node_features_for_atlas.progress_apply(lambda x: x['betweenness_centrality'])

    # Convert the node-based features to a list
    for atlas in ['cc200', 'aal', 'dos160']:
        for feature in ['degree', 'degree_centrality', 'eigenvector_centrality', 'closeness_centrality', 'betweenness_centrality']:
            node_based_features[f'{atlas}_{feature}'] = node_based_features[f'{atlas}_{feature}'].progress_apply(list)

    return node_based_features


def edge_betweenness_centrality(nx_graph: nx.Graph, edge_weight_threshold: Union[float, None] = None) -> List[float]:
    filtered_graph = nx_graph
    if edge_weight_threshold is not None:
        filtered_graph = graph_generation.apply_edge_weight_threshold(nx_graph, edge_weight_threshold)
    edge_betweenness = nx.edge_betweenness_centrality(filtered_graph)
    return list(edge_betweenness.values())

def edge_density(nx_graph: nx.Graph, edge_weight_threshold: Union[float, None] = None) -> List[float]:
    filtered_graph = nx_graph
    if edge_weight_threshold is not None:
        filtered_graph = graph_generation.apply_edge_weight_threshold(nx_graph, edge_weight_threshold)
    densities = [nx.density(filtered_graph.subgraph(c)) for c in nx.connected_components(filtered_graph)]
    return densities

def average_degree(nx_graph: nx.Graph, edge_weight_threshold: Union[float, None] = None) -> List[float]:
    filtered_graph = nx_graph
    if edge_weight_threshold is not None:
        filtered_graph = graph_generation.apply_edge_weight_threshold(nx_graph, edge_weight_threshold)
    average_degrees = [sum(d for _, d in filtered_graph.degree(c)) / len(c) for c in nx.connected_components(filtered_graph)] # type: ignore
    return average_degrees

def clustering_coefficient(nx_graph: nx.Graph, edge_weight_threshold: Union[float, None] = None) -> List[float]:
    filtered_graph = nx_graph
    if edge_weight_threshold is not None:
        filtered_graph = graph_generation.apply_edge_weight_threshold(nx_graph, edge_weight_threshold)
    clustering_coefficients = [nx.average_clustering(filtered_graph.subgraph(c)) for c in nx.connected_components(filtered_graph)]
    return clustering_coefficients

def number_of_connected_components(nx_graph: nx.Graph, edge_weight_threshold: Union[float, None] = None) -> int:
    filtered_graph = nx_graph
    if edge_weight_threshold is not None:
        filtered_graph = graph_generation.apply_edge_weight_threshold(nx_graph, edge_weight_threshold)
    return len(list(nx.connected_components(filtered_graph)))

def average_shortest_path_length(nx_graph: nx.Graph, edge_weight_threshold: Union[float, None] = None) -> List[float]:
    filtered_graph = nx_graph
    if edge_weight_threshold is not None:
        filtered_graph = graph_generation.apply_edge_weight_threshold(nx_graph, edge_weight_threshold)
    path_lengths = [nx.average_shortest_path_length(filtered_graph.subgraph(c)) for c in nx.connected_components(filtered_graph)]
    return path_lengths

# Define a function to compute edge-based features
def edge_features(nx_graph: nx.Graph, edge_weight_threshold: Union[float, None] = None) -> dict[str, Union[List[float], int]]:
    filtered_graph = nx_graph
    if edge_weight_threshold is not None:
        filtered_graph = graph_generation.apply_edge_weight_threshold(nx_graph, edge_weight_threshold)
    return {
        'edge_betweenness_centrality': edge_betweenness_centrality(filtered_graph),
        'edge_density': edge_density(filtered_graph),
        'average_degree': average_degree(filtered_graph),
        'clustering_coefficient': clustering_coefficient(filtered_graph),
        'number_of_connected_components': number_of_connected_components(filtered_graph),
        'average_shortest_path_length': average_shortest_path_length(filtered_graph),
    }

def compute_edge_based_features(df_subjects: pd.DataFrame, edge_weight_threshold: float) -> pd.DataFrame:
    # Enable tqdm integration with Pandas
    tqdm.pandas()

    # Apply the function to each graph and store the results in a new dataframe
    edge_based_features = df_subjects[['id', 'ASD', 'SITE_ID']].copy()

    for atlas in tqdm(['cc200', 'aal', 'dos160'], desc='Computing edge-based features'):
        edge_features_for_atlas = df_subjects[f'nx_graph_{atlas}'].progress_apply(edge_features, args=(edge_weight_threshold,))

        edge_based_features[f'{atlas}_edge_betweenness_centrality'] = edge_features_for_atlas.progress_apply(lambda x: x['edge_betweenness_centrality'])
        edge_based_features[f'{atlas}_edge_density'] = edge_features_for_atlas.progress_apply(lambda x: x['edge_density'])
        edge_based_features[f'{atlas}_average_degree'] = edge_features_for_atlas.progress_apply(lambda x: x['average_degree'])
        edge_based_features[f'{atlas}_clustering_coefficient'] = edge_features_for_atlas.progress_apply(lambda x: x['clustering_coefficient'])
        edge_based_features[f'{atlas}_number_of_connected_components'] = edge_features_for_atlas.progress_apply(lambda x: x['number_of_connected_components'])
        edge_based_features[f'{atlas}_average_shortest_path_length'] = edge_features_for_atlas.progress_apply(lambda x: x['average_shortest_path_length'])

    # Convert the edge-based features to a list
    for atlas in ['cc200', 'aal', 'dos160']:
        for feature in ['edge_betweenness_centrality', 'edge_density', 'average_degree', 'clustering_coefficient', 'average_shortest_path_length']:
            edge_based_features[f'{atlas}_{feature}'] = edge_based_features[f'{atlas}_{feature}'].progress_apply(list)

    return edge_based_features


def compute_diameter(nx_graph: nx.Graph) -> float:
    components = list(nx.connected_components(nx_graph))
    diameters = [nx.diameter(nx_graph.subgraph(c)) for c in components]
    return max(diameters)

def compute_radius(nx_graph: nx.Graph) -> float:
    components = list(nx.connected_components(nx_graph))
    radii = [nx.radius(nx_graph.subgraph(c)) for c in components]
    return max(radii)

# Define a function to compute graph-level features
def graph_features(nx_graph: nx.Graph, edge_weight_threshold: Union[float, None] = None) -> dict[str, Union[int, float, Mapping[Any, Any]]]:
    filtered_graph = nx_graph
    if edge_weight_threshold is not None:
        filtered_graph = graph_generation.apply_edge_weight_threshold(nx_graph, edge_weight_threshold)
    return {
        'number_of_nodes': filtered_graph.number_of_nodes(),
        'number_of_edges': filtered_graph.number_of_edges(),
        'average_degree': nx.average_degree_connectivity(filtered_graph),
        'density': nx.density(filtered_graph),
        'diameter': compute_diameter(filtered_graph),
        'radius': compute_radius(filtered_graph),
        'assortativity_coefficient': nx.degree_assortativity_coefficient(filtered_graph),
        'transitivity': nx.transitivity(filtered_graph),
        'average_clustering': nx.average_clustering(filtered_graph),
        # 'modularity': nx.community.modularity(filtered_graph),
    }

def compute_graph_level_features(df_subjects: pd.DataFrame, edge_weight_threshold: float) -> pd.DataFrame:
    # Enable tqdm integration with Pandas
    tqdm.pandas()

    # Apply the function to each graph and store the results in a new dataframe
    graph_level_features = df_subjects[['id', 'ASD', 'SITE_ID']].copy()

    for atlas in tqdm(['cc200', 'aal', 'dos160'], desc='Computing graph-level features'):
        graph_features_for_atlas = df_subjects[f'nx_graph_{atlas}'].progress_apply(graph_features, args=(edge_weight_threshold,))

        graph_level_features[f'{atlas}_number_of_nodes'] = graph_features_for_atlas.progress_apply(lambda x: x['number_of_nodes'])
        graph_level_features[f'{atlas}_number_of_edges'] = graph_features_for_atlas.progress_apply(lambda x: x['number_of_edges'])
        graph_level_features[f'{atlas}_average_degree'] = graph_features_for_atlas.progress_apply(lambda x: x['average_degree'])
        graph_level_features[f'{atlas}_density'] = graph_features_for_atlas.progress_apply(lambda x: x['density'])
        graph_level_features[f'{atlas}_diameter'] = graph_features_for_atlas.progress_apply(lambda x: x['diameter'])
        graph_level_features[f'{atlas}_radius'] = graph_features_for_atlas.progress_apply(lambda x: x['radius'])
        graph_level_features[f'{atlas}_assortativity_coefficient'] = graph_features_for_atlas.progress_apply(lambda x: x['assortativity_coefficient'])
        graph_level_features[f'{atlas}_transitivity'] = graph_features_for_atlas.progress_apply(lambda x: x['transitivity'])
        graph_level_features[f'{atlas}_average_clustering'] = graph_features_for_atlas.progress_apply(lambda x: x['average_clustering'])
        # graph_level_features[f'{atlas}_modularity'] = graph_features_for_atlas.progress_apply(lambda x: x['modularity'])

    return graph_level_features