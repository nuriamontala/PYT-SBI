import torch
import networkx as nx
from torch_geometric.data import Data
import pandas as pd
from graphein.protein.config import ProteinGraphConfig
from graphein.protein.graphs import construct_graph

def load_residue_features(csv_path):
    """
    Reads a CSV file containing residue features and binding site labels.
    Residue IDs are stored as tuples (residue_number, chain).
    
    Returns:
    - features_dict: Dictionary {(residue_number, chain): feature_vector}
    - labels_dict: Dictionary {(residue_number, chain): binding_site_label}
    """
    df = pd.read_csv(csv_path)

    # Convert first column (Residue ID) into tuples (residue_number, chain)
    df.iloc[:, 0] = df.iloc[:, 0].apply(eval)  # Convert string "(1, 'L')" → tuple (1, 'L')

    residue_ids = df.iloc[:, 0]  # Residue ID as tuple
    features = df.iloc[:, 1:-1].values  # All feature columns except last
    labels = df.iloc[:, -1].values  # Last column as binding site labels

    # Create lookup dictionaries
    features_dict = {res_id: feat for res_id, feat in zip(residue_ids, features)}
    labels_dict = {res_id: label for res_id, label in zip(residue_ids, labels)}

    return features_dict, labels_dict

def networkx_to_pyg(G_nx, features_dict, labels_dict, num_features):
    """
    Converts a Graphein-generated NetworkX protein graph to a PyTorch Geometric Data object.
    - Uses external residue features from CSV.
    - Uses last column from CSV for node labels (binding sites).
    - Correctly handles (residue_number, chain) format.
    """
    node_map = {}  # Maps node (residue, chain) to index
    reverse_map = {}  # Reverse lookup: PyG index → (residue_number, chain)
    node_features = []
    y = []

    # Convert nodes from Graphein's graph
    for i, (node, attr) in enumerate(G_nx.nodes(data=True)):
        res_id = (attr.get("residue_number"), attr.get("chain_id"))  # Standardized format

        node_map[node] = i  # Assign PyG-compatible node index
        reverse_map[i] = res_id  # Store mapping back to residue identifier

        # Retrieve features (default to zero vector if missing)
        features = features_dict.get(res_id, [0] * num_features)
        node_features.append(features)

        # Retrieve binding site labels (default = non-binding)
        y.append(labels_dict.get(res_id, 0))

    x = torch.tensor(node_features, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.long)

    # Extract edges
    edges = []
    for u, v in G_nx.edges():
        try:
            edges.append((node_map[u], node_map[v]))  # Use fixed node IDs
        except KeyError:
            print(f"⚠️ Skipping edge ({u}, {v}) due to missing node mapping!")

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    return Data(x=x, edge_index=edge_index, y=y), reverse_map

# Paths
pdb_path = "1fig.pdb"
csv_path = "1fig.csv"

# Load residue features & labels (with chain information)
features_dict, labels_dict = load_residue_features(csv_path)

# Construct the protein graph using Graphein
config = ProteinGraphConfig()
G_nx = construct_graph(config=config, path=pdb_path)

# Convert Graph and Get Residue Mapping
protein_graph_data, reverse_map = networkx_to_pyg(G_nx, features_dict, labels_dict, num_features)

# Print Feature Vectors with Associated Residues
print("\n🔹 First 10 Residue Features & Corresponding Nodes 🔹")
for node_idx in range(min(10, protein_graph_data.x.shape[0])):  # First 10 nodes
    residue_id = reverse_map.get(node_idx, "Unknown")  # Get residue number & chain
    features = protein_graph_data.x[node_idx].tolist()  # Convert to list for readability
    label = protein_graph_data.y[node_idx].item()  # Get binding site label

    print(f"Node {node_idx}: Residue {residue_id} | Label: {label} | Features: {features}\n")
