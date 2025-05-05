import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

caltech_graph = nx.read_gml('data/Caltech36.gml', label='id')
mit_graph = nx.read_gml('data/MIT8.gml', label='id')
johns_hopkins_graph = nx.read_gml('data/Johns_Hopkins55.gml', label='id')

graphs = {
    "Caltech": caltech_graph,
    "MIT": mit_graph,
    "Johns Hopkins": johns_hopkins_graph
}

for name, graph in graphs.items():
    # print(f"Plotting {name} graph...")
    # plt.figure(figsize=(10, 6))
    # pos = nx.spring_layout(graph)
    # nx.draw(graph, pos, cmap=plt.cm.viridis, node_size=50, alpha=0.7)
    # plt.title(f"{name} Graph")
    # plt.show()
    degree_sequence = sorted([d for n, d in graph.degree()], reverse=True)
    plt.figure(figsize=(10, 6))
    plt.hist(degree_sequence, bins=30, color='blue', alpha=0.7)
    plt.title(f"{name} Graph Degree Distribution")
    plt.xlabel("Degree")
    plt.ylabel("Frequency")
    plt.grid()
    plt.show()

    global_clustering = nx.transitivity(graph)
    mean_local_clustering = np.mean(list(nx.clustering(graph).values()))
    print(f"{name} Graph:")
    print(f"  Global Clustering Coefficient: {global_clustering:.4f}")
    print(f"  Mean Local Clustering Coefficient: {mean_local_clustering:.4f}")
    print()

    edge_density = nx.density(graph)
    print(f"  Edge Density: {edge_density:.4f}")
    print()

    node_degree = dict(graph.degree())
    node_degree = [node_degree[node] for node in graph.nodes()]
    clustering = nx.clustering(graph)
    node_clustering = [clustering[node] for node in graph.nodes()]
    plt.figure(figsize=(10, 6))
    plt.scatter(node_degree, node_clustering, alpha=0.5)
    plt.title(f"{name} Graph: Degree vs Local Clustering Coefficient")
    plt.xlabel("Degree")
    plt.ylabel("Local Clustering Coefficient")
    plt.grid()
    plt.show()
