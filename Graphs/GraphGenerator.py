import torch
import os
import ast
import numpy as np
import pandas as pd
from torch_geometric.data import Data
from graphein.protein.config import ProteinGraphConfig
from graphein.protein.config import ProteinGraphConfig
from graphein.protein.graphs import construct_graph
from graphein.protein.edges.distance import (add_peptide_bonds,
                                             add_hydrogen_bond_interactions,
                                             add_disulfide_interactions,                                       
                                             add_ionic_interactions,
                                             add_aromatic_interactions,
                                             add_aromatic_sulphur_interactions,
                                             add_cation_pi_interactions,
                                             add_delaunay_triangulation)

# Edge construction functions
new_edge_funcs = {"edge_construction_functions": [
    add_peptide_bonds,
    add_aromatic_interactions,
    add_hydrogen_bond_interactions,
    add_disulfide_interactions,
    add_ionic_interactions,
    add_aromatic_sulphur_interactions,
    add_cation_pi_interactions,
    add_delaunay_triangulation
]}

# 3-letter to 1-letter amino acid code mapping
three_to_one = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D',
    'CYS': 'C', 'GLU': 'E', 'GLN': 'Q', 'GLY': 'G',
    'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
    'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S',
    'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
    'SEC': 'U', 'PYL': 'O', 'ASX': 'B', 'GLX': 'Z',
    'XLE': 'J', 'UNK': 'X'
}


# Load Residue Features from CSV
def load_residue_features(csv_path):
    """Loads residue features and binding site labels from CSV."""
    df = pd.read_csv(csv_path, sep='\t')

    # Convert first column (Residue_ID) into tuples safely
    df.iloc[:, 0] = df.iloc[:, 0].apply(ast.literal_eval) # Converts to tuple (num, chain, res)

    residue_ids = df.iloc[:, 0]  # Residue ID as tuple
    features = df.iloc[:, 1:-1].values  # Feature columns (excluding label)
    labels = df.iloc[:, -1].values  # Last column = binding site labels

    # Create lookup dictionaries
    features_dict = {res_id: feat for res_id, feat in zip(residue_ids, features)}
    labels_dict = {res_id: label for res_id, label in zip(residue_ids, labels)}

    return features_dict, labels_dict, features.shape[1]  # Return num_features

# Convert NetworkX Graph → PyTorch Geometric Graph
def networkx_to_pyg(G_nx, features_dict, labels_dict, num_features):
    """Converts a NetworkX protein graph to a PyTorch Geometric Data object, removing nodes without features."""
    node_map = {}  # Maps node (residue, chain) to index
    reverse_map = {}  # Reverse lookup: PyG index → (residue_number, chain)
    node_features = []
    y = []
    valid_nodes = set()  # Stores nodes that have valid features

    for i, (node, attr) in enumerate(G_nx.nodes(data=True)):
        res_name_3 = attr.get("residue_name")
        res_name_1 = three_to_one.get(res_name_3, 'X')  # Use 'X' as fallback for unknown residues
        res_id = (attr.get("residue_number"), attr.get("chain_id"), res_name_1)
                
        # Only include nodes that have valid features
        if res_id not in features_dict:
            print(f"⚠️ Warning: Residue {res_id} in graph but missing from CSV! Skipping.")
            continue  # Skip nodes with missing features

        valid_nodes.add(node)
        node_map[node] = len(node_features)  # Assign PyG-compatible node index
        reverse_map[len(node_features)] = res_id  # Store mapping back to residue identifier

        # Retrieve features
        node_features.append(features_dict[res_id])

        # Retrieve binding site labels
        y.append(labels_dict.get(res_id, 0))

    x = torch.tensor(np.array(node_features), dtype=torch.float)
    y = torch.tensor(y, dtype=torch.long)

    # Extract edges but only keep edges where **both** nodes are in valid_nodes
    edges = []
    for u, v in G_nx.edges():
        if u in valid_nodes and v in valid_nodes:  # Ensure both nodes exist in the filtered set
            edges.append((node_map[u], node_map[v]))

    if not edges:  # If no valid edges, return None (to avoid empty graphs)
        print(f"⚠️ Warning: Graph contains no valid edges after filtering. Skipping!")
        return None, None

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    return Data(x=x, edge_index=edge_index, y=y), reverse_map

# Process All PDB + CSV Files in a Folder
def process_protein_graphs(folder_path, save_path="../files/protein_graphs", start=0, end=500):
    
    """Processes PDB files in batches of 500 and saves graphs with PDB codes as filenames."""
    
    os.makedirs(save_path, exist_ok=True)  # Create output folder if it doesn't exist

    # Get all available PDB files
    pdb_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".pdb")])  # Sort for consistency

    # Select a subset (batch of 500)
    pdb_files = pdb_files[start:end]

    for pdb_file in pdb_files:
        pdb_code = pdb_file.replace(".pdb", "")  # Extract PDB code
        pdb_path = os.path.join(folder_path, pdb_file)
        csv_path = os.path.join(folder_path, pdb_code + ".csv")
        graph_file = os.path.join(save_path, f"{pdb_code}.pt")  # Use PDB code as filename

        # Skip if the graph file already exists (avoid overwriting)
        if os.path.exists(graph_file):
            print(f"⚠️ Warning: Graph for {pdb_code} already exists. Skipping!")
            continue

        if not os.path.exists(csv_path):
            print(f"⚠️ Warning: No CSV found for {pdb_file}. Skipping!")
            continue

        print(f"Processing {pdb_file}...")

        # Load features & labels
        features_dict, labels_dict, num_features = load_residue_features(csv_path)

        # Construct NetworkX graph
        config = ProteinGraphConfig(**new_edge_funcs)
        G_nx = construct_graph(config=config, path=pdb_path)

        # Convert to PyG Data object
        protein_graph_data, reverse_map = networkx_to_pyg(G_nx, features_dict, labels_dict, num_features)

        if protein_graph_data:  # Ensure we have a valid graph
            torch.save(protein_graph_data, graph_file)
            print(f"Saved graph as {graph_file}")

    print(f"Batch {start}-{end} processed and saved in {save_path}")

# Run
folder_path = "../files/graphs"
process_protein_graphs(folder_path, start=14500, end=15500)
