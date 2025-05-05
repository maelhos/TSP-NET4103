import networkx as nx
import numpy as np
import torch
from sklearn.metrics import accuracy_score, mean_absolute_error

def label_propagation(graph, labels, max_iter=100):
    nodes = list(graph.nodes())
    n = len(nodes)
    adj_matrix = nx.to_numpy_array(graph)
    adj_matrix = torch.tensor(adj_matrix, dtype=torch.float32)

    unique_labels = {label: idx for idx, label in enumerate(set(labels.values()))}
    label_matrix = torch.zeros((n, len(unique_labels)))
    for node, label in labels.items():
        label_matrix[node, unique_labels[label]] = 1

    degree_matrix = torch.diag(torch.sum(adj_matrix, dim=1))
    norm_adj_matrix = torch.linalg.inv(degree_matrix) @ adj_matrix

    for _ in range(max_iter):
        label_matrix = norm_adj_matrix @ label_matrix
        label_matrix = torch.nn.functional.normalize(label_matrix, p=1, dim=1)

    propagated_labels = {node: torch.argmax(label_matrix[node]).item() for node in range(n)}
    return propagated_labels

def evaluate_label_propagation(graph, attribute, missing_ratios):
    original_labels = nx.get_node_attributes(graph, attribute)
    nodes = list(graph.nodes())

    for ratio in missing_ratios:
        num_missing = int(ratio * len(nodes))
        missing_nodes = np.random.choice(nodes, num_missing, replace=False)
        observed_labels = {node: label for node, label in original_labels.items() if node not in missing_nodes}

        propagated_labels = label_propagation(graph, observed_labels)

        true_labels = [original_labels[node] for node in missing_nodes]
        pred_labels = [propagated_labels[node] for node in missing_nodes]
        accuracy = accuracy_score(true_labels, pred_labels)
        mae = mean_absolute_error(true_labels, pred_labels)

        print(f"Missing Ratio: {ratio}")
        print(f"Accuracy: {accuracy:.4f}, Mean Absolute Error: {mae:.4f}")

caltech_graph = nx.read_gml('data/Caltech36.gml', label='id')
mit_graph = nx.read_gml('data/MIT8.gml', label='id')
johns_hopkins_graph = nx.read_gml('data/Johns_Hopkins55.gml', label='id')

graphs = {
    "Caltech": caltech_graph,
    "MIT": mit_graph,
    "Johns Hopkins": johns_hopkins_graph
}

if __name__ == "__main__":
    for name, graph in graphs.items():
        print(f"Processing {name} graph...")
        nx.set_node_attributes(graph, {node: np.random.randint(0, 3) for node in graph.nodes()}, "dorm")

        for attribute in ["dorm", "major_index", "gender"]:
            print(f"Evaluating label propagation for {name} graph, attribute: {attribute}")
            evaluate_label_propagation(graph, attribute, [0.1, 0.2, 0.3])
            print("-" * 50)