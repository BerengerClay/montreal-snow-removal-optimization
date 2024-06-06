import geopandas as gpd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import itertools

def load_and_prepare_data(geojson_fp, quartiers_interet):
    gdf = gpd.read_file(geojson_fp)

    if gdf.crs is None:
        gdf.set_crs(epsg=4326, inplace=True)

    gdf_filtered = gdf[(gdf['ARR_GCH'].isin(quartiers_interet)) | 
                       (gdf['ARR_DRT'].isin(quartiers_interet))]
    
    return gdf_filtered

def build_graph_from_gdf(gdf):
    G = nx.DiGraph()

    for _, row in gdf.iterrows():
        coords = list(row.geometry.coords)
        sens_cir = row['SENS_CIR']
        for i in range(len(coords) - 1):
            u = (coords[i][1], coords[i][0])
            v = (coords[i + 1][1], coords[i + 1][0])
            weight = np.linalg.norm(np.array(u) - np.array(v))
            if sens_cir == 1:
                G.add_edge(u, v, weight=weight, direction='one-way')
            elif sens_cir == -1:
                G.add_edge(v, u, weight=weight, direction='one-way')
            else:
                G.add_edge(u, v, weight=weight, direction='two-way')
                G.add_edge(v, u, weight=weight, direction='two-way')
    
    return G

def plot_graph_with_components(G, title, filename):
    pos = {node: (node[1], node[0]) for node in G.nodes()}
    fig, ax = plt.subplots()

    # Find strongly connected components
    scc = list(nx.strongly_connected_components(G))
    
    # Generate colors
    colors = itertools.cycle(plt.cm.get_cmap('tab20').colors)
    
    # Track drawn edges and nodes
    drawn_edges = set()
    drawn_nodes = set()

    # Draw components and edges with specific colors
    for component in scc:
        subgraph = G.subgraph(component)
        if len(component) >= 10:
            color = next(colors)
            nx.draw_networkx_nodes(subgraph, pos, node_color=color, node_size=0.5, ax=ax)
            nx.draw_networkx_edges(subgraph, pos, edge_color=color, width=0.5, ax=ax, arrows=False)
            
            for u, v, d in subgraph.edges(data=True):
                drawn_edges.add((u, v))
                if d['direction'] == 'one-way':
                    nx.draw_networkx_edges(subgraph, pos, edgelist=[(u, v)], edge_color=color, width=0.5, ax=ax, arrows=True,
                                           arrowstyle='-|>', arrowsize=5)
            
            drawn_nodes.update(component)

    # Draw missing edges and nodes in red
    all_edges = set(G.edges())
    missing_edges = all_edges - drawn_edges
    nx.draw_networkx_edges(G, pos, edgelist=missing_edges, edge_color='red', width=0.5, ax=ax, arrows=True,
                           arrowstyle='-|>', arrowsize=5)
    nx.draw_networkx_edges(G, pos, edgelist=missing_edges, edge_color='red', width=0.5, ax=ax, arrows=False)

    all_nodes = set(G.nodes())
    missing_nodes = all_nodes - drawn_nodes
    nx.draw_networkx_nodes(G, pos, nodelist=missing_nodes, node_color='red', node_size=0.5, ax=ax)

    plt.title(title)
    plt.savefig(filename, format='png', dpi=300)  # Save the figure as PNG with high resolution
    plt.close()  # Close the figure to free memory

def main():
    geojson_fp = 'Data/geobase.json'
    quartiers_interet = ['Outremont']
    
    gdf_filtered = load_and_prepare_data(geojson_fp, quartiers_interet)
    
    for quartier in quartiers_interet:
        gdf_quartier = gdf_filtered[(gdf_filtered['ARR_GCH'] == quartier) | (gdf_filtered['ARR_DRT'] == quartier)]
        G = build_graph_from_gdf(gdf_quartier)
        plot_graph_with_components(G, f'{quartier} - Strongly Connected Components', f'{quartier}_graph.png')

if __name__ == "__main__":
    main()
