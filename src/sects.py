import networkx as nx
import pandas as pd
from collections import Counter

caltech_graph = nx.read_gml('data/Caltech36.gml', label='id')
mit_graph = nx.read_gml('data/MIT8.gml', label='id')
johns_hopkins_graph = nx.read_gml('data/Johns_Hopkins55.gml', label='id')

graphs = {
    "Caltech": caltech_graph,
    "MIT": mit_graph,
    "Johns Hopkins": johns_hopkins_graph
}

def get_dominant_trait(graph, nodes):
    attrs = ['dorm', 'year', 'major_index']
    best_trait = "None"
    best_purity = 0

    for attr in attrs:
        values = [graph.nodes[n].get(attr) for n in nodes if graph.nodes[n].get(attr) != 0]
        if not values: 
            continue
        
        counts = Counter(values)
        most_common, count = counts.most_common(1)[0]
        purity = count / len(values) 
        
        if purity > best_purity:
            best_purity = purity
            best_trait = f"{attr}={most_common}"
            
    return best_trait, best_purity


for idx, (name, graph) in enumerate(graphs.items()):
    print(f"Processing {name} graph...")
    
    # Here I chose louvain_communities as it seemed to work better in this case
    comms = nx.community.louvain_communities(graph, resolution = 2)
    
    # The "gated communityness" is actually the conductance
    gated_data = []
    for c in comms:
        if len(c) < 5: continue # Skip tiny groups overwise results are biased
        cond = nx.conductance(graph, c)
        trait, purity = get_dominant_trait(graph, c)
        gated_data.append({'conductance': cond, 'size': len(c), 'trait': trait, 'purity': purity})

    df = pd.DataFrame(gated_data).sort_values('conductance')

    print(f"Total communities found: {len(df)}")
    print("Top 3 sects (Lowest Conductance):")
    print(df.head(3)[['conductance', 'size', 'trait', 'purity']].to_string(index=False))

    thresholds = [0.3, 0.4, 0.5, 0.6]
    print(f"Count of Gated Groups by Conductance Threshold:")
    for t in thresholds:
        count = len(df[df['conductance'] < t])
        print(f"  < {t}: {count} groups")
