import networkx as nx
import numpy as np
from abc import ABC, abstractmethod
import random

class LinkPrediction(ABC):
    def __init__(self, graph):
        self.graph = graph
        self.N = len(graph)
    
    def neighbors(self, v):
        neighbors_list = self.graph.neighbors(v)
        return list(neighbors_list)

    @abstractmethod
    def fit(self):
        raise NotImplementedError("Fit must be implemented")

class CommonNeighbors(LinkPrediction):
    def __init__(self, graph):
        super(CommonNeighbors, self).__init__(graph)
        
    def fit(self):
        self.scores = {}
        for u in self.graph.nodes():
            for v in self.graph.nodes():
                if u != v and not self.graph.has_edge(u, v):
                    self.scores[(u, v)] = len(set(self.neighbors(u)).intersection(self.neighbors(v)))

class Jaccard(LinkPrediction):
    def __init__(self, graph):
        super(Jaccard, self).__init__(graph)

    def fit(self):
        self.scores = {}
        for u in self.graph.nodes():
            for v in self.graph.nodes():
                if u != v and not self.graph.has_edge(u, v):
                    intersection = len(set(self.neighbors(u)).intersection(self.neighbors(v)))
                    union = len(set(self.neighbors(u)).union(self.neighbors(v)))
                    self.scores[(u, v)] = intersection / union if union != 0 else 0

class AdamicAdar(LinkPrediction):
    def __init__(self, graph):
        super(AdamicAdar, self).__init__(graph)

    def fit(self):
        self.scores = {}
        for u in self.graph.nodes():
            for v in self.graph.nodes():
                if u != v and not self.graph.has_edge(u, v):
                    common_neighbors = set(self.neighbors(u)).intersection(self.neighbors(v))
                    score = sum(1 / np.log(len(self.neighbors(w))) for w in common_neighbors if len(self.neighbors(w)) > 1)
                    self.scores[(u, v)] = score

caltech_graph = nx.read_gml('data/Caltech36.gml', label='id')
mit_graph = nx.read_gml('data/MIT8.gml', label='id')
johns_hopkins_graph = nx.read_gml('data/Johns_Hopkins55.gml', label='id')

graphs = {
    "Caltech": caltech_graph,
    "MIT": mit_graph,
    "Johns Hopkins": johns_hopkins_graph
}

def evaluate_predictor(predictor: LinkPrediction, graph: nx.Graph, removed_fractions: list, k_values: list):
    results_list = []
    for removed_fraction in removed_fractions:
        edges = list(graph.edges())
        num_edges_to_remove = int(len(edges) * removed_fraction)
        removed_edges = random.sample(edges, num_edges_to_remove)
        
        test_graph = graph.copy()
        test_graph.remove_edges_from(removed_edges)
        
        predictor.graph = test_graph
        predictor.fit()
        scores = sorted(predictor.scores.items(), key=lambda x: x[1], reverse=True)
        
        results = {}
        for k in k_values:
            top_k_pairs = [pair for pair, _ in scores[:k]]
            predicted_edges = set(top_k_pairs)
            true_edges = set(removed_edges)
            
            tp = len(set(predicted_edges) & set(true_edges))
            fp = len(set(predicted_edges) - set(true_edges))
            fn = len(set(true_edges) - set(predicted_edges))
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            results[k] = {
                "precision": precision,
                "recall": recall,
                "top@k": tp / k if k > 0 else 0
            }
        results_list.append(results)
    return results_list

for name, graph in graphs.items():
    common_neighbors = CommonNeighbors(graph)
    jaccard = Jaccard(graph)
    adamic_adar = AdamicAdar(graph)

    removed_fractions = [0.05, 0.1, 0.15, 0.2]
    k_values = [50, 100, 200, 400]

    print(f"Evaluating on {name} graph:")
    for predictor, predictor_name in [(common_neighbors, "Common Neighbors"),
                                       (adamic_adar, "Adamic-Adar"), 
                                       (jaccard, "Jaccard")]:
        results_list = evaluate_predictor(predictor, graph.copy(), removed_fractions, k_values)
        print(f"  {predictor_name}:")
        for k in k_values:
            for frac in removed_fractions:
                results = results_list[removed_fractions.index(frac)]
                precision = results[k]["precision"]
                recall = results[k]["recall"]
                top_k = results[k]["top@k"]
                print(f"    k={k}, removed_fraction={frac}: precision={precision:.4f}, recall={recall:.4f}, top@k={top_k:.4f}")
    print()
