import networkx as nx
import matplotlib.pyplot as plt
import random

def find_hubs(graph, k, n):
    nodes = list(graph.nodes())
    score =  {node: 0 for node in nodes}

    for _ in range(k):
        n1, n2 = random.choice(nodes), random.choice(nodes)
        try:
            paths = nx.all_shortest_paths(graph, n1, n2)
         
            for path in paths:
                for node in path[1:-1]: score[node] += 1
        except: # if no path found
            continue

    top_scores = sorted(score.items(), key=lambda x: x[1])
    return list(map(lambda x: x[0], top_scores[-n:]))

def find_highest_degrees(graph, n):
    sorted_degs = sorted(list(graph.nodes()), key=lambda x: graph.degree(x))
    return sorted_degs[-n:]

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
        for n in (10, 100, 1000):
            hdgs = find_highest_degrees(graph, n)
            for k in (200, 800, 1600):
                hubs = find_hubs(graph, k, n)

                print(f"For k={k}, n={n}, correlation is :", len(set(hdgs).intersection(set(hubs))) / n)
