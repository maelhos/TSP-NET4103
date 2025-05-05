import networkx as nx
import matplotlib.pyplot as plt
import os

DATA_DIR = "data"
OUTPUT_DIR = "plots"

os.makedirs(OUTPUT_DIR, exist_ok=True)
graph_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.gml')]

graphs = {}
for file in graph_files:
    graph_name = os.path.splitext(file)[0]
    graphs[graph_name] = nx.read_gml(os.path.join(DATA_DIR, file), label='id')

labels = ["student_fac", "major_index", "dorm", "gender", "degree", "size"]
metrics = {label: [] for label in labels}

for name, graph in graphs.items():
    print(f"Processing {name} graph...")
    
    for label in labels:
        if label not in ["degree", "size"]:
            if label in nx.get_node_attributes(graph, label):
                metrics[label].append(nx.attribute_assortativity_coefficient(graph, label))
            else:
                metrics[label].append(None)
        elif label == "degree":
            metrics[label].append(nx.degree_assortativity_coefficient(graph))
        else:
            metrics["size"].append(len(graph.nodes))

label_title = {
    "student_fac": "Faculty status",
    "major_index": "Major",
    "degree": "Vertex degree",
    "gender": "Gender",
    "dorm": "Dorm"
}

for label in labels:
    if label != "size" and metrics[label]:
        valid_data = [(size, value) for size, value in zip(metrics["size"], metrics[label]) if value is not None]
        if not valid_data:
            continue
        sizes, values = zip(*valid_data)
        
        # Scatter plot
        plt.style.use('seaborn-v0_8-darkgrid')
        plt.figure(figsize=(8, 6))
        plt.scatter(sizes, values, c='blue', edgecolor='k', alpha=0.7)
        plt.xlabel("Network size")
        plt.ylabel(f"{label_title[label]} assortativity")
        plt.xscale("log")
        plt.title(f"Scatter Plot: {label_title[label]} Assortativity vs Network Size")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'scatter_assortativity_{label}.png'))
        plt.close()
        
        # Histogram
        plt.style.use('seaborn-v0_8-darkgrid')
        plt.figure(figsize=(8, 6))
        plt.hist(values, bins=20, density=True, color='green', edgecolor='black', alpha=0.7)
        plt.xlabel(f"{label_title[label]} assortativity")
        plt.ylabel("Density")
        plt.title(f"Histogram: {label_title[label]} Assortativity")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'histogram_assortativity_{label}.png'))
        plt.close()
