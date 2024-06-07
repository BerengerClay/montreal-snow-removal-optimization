import geopandas as gpd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from geopy.distance import geodesic

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
        street_name = row['NOM_VOIE'] if row['NOM_VOIE'] else 'Unknown Street'
        for i in range(len(coords) - 1):
            u = (coords[i][1], coords[i][0])
            v = (coords[i + 1][1], coords[i + 1][0])
            weight = geodesic(u, v).meters
            if sens_cir == 1:
                G.add_edge(u, v, weight=weight, direction='one-way', street_name=street_name)
            elif sens_cir == -1:
                G.add_edge(v, u, weight=weight, direction='one-way', street_name=street_name)
            else:
                G.add_edge(u, v, weight=weight, direction='two-way', street_name=street_name)
                G.add_edge(v, u, weight=weight, direction='two-way', street_name=street_name)
    return G

def find_eulerian_path(G, start_node=None):
    if not nx.is_eulerian(G):
        odd_degree_nodes = [v for v, d in G.degree() if d % 2 == 1]
        G.add_edge(odd_degree_nodes[0], odd_degree_nodes[1], weight=0)

    if start_node:
        return list(nx.eulerian_circuit(G, source=start_node))
    else:
        return list(nx.eulerian_circuit(G))

def plot_graph_with_subgraph(G, subgraph_nodes, title, filename):
    pos = {node: (node[1], node[0]) for node in G.nodes()}
    fig, ax = plt.subplots(figsize=(10, 8))

    for u, v, d in G.edges(data=True):
        if d['direction'] != 'two-way':
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], edge_color='gray', width=1, ax=ax, arrows=True,
                                   arrowstyle='-|>', arrowsize=10)

    for u, v, d in G.edges(data=True):
        if d['direction'] == 'two-way' and (v, u) in G.edges:
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], edge_color='gray', width=1, ax=ax, arrows=False)

    subgraph = G.subgraph(subgraph_nodes)
    for u, v, d in subgraph.edges(data=True):
        if d['direction'] == 'two-way' and (v, u) in subgraph.edges:
            nx.draw_networkx_edges(subgraph, pos, edgelist=[(u, v)], edge_color='green', width=1.5, ax=ax, arrows=False)
        else:
            nx.draw_networkx_edges(subgraph, pos, edgelist=[(u, v)], edge_color='green', width=1.5, ax=ax, arrows=True,
                                   arrowstyle='-|>', arrowsize=10)

    plt.title(title)
    plt.savefig(filename, format='png', dpi=300)
    plt.close()

def animate_cpp(G, path, filename):
    pos = {node: (node[1], node[0]) for node in G.nodes()}
    fig, ax = plt.subplots(figsize=(10, 8))

    def update(num):
        ax.clear()
        nx.draw(G, pos, with_labels=False, node_color='lightblue', edge_color='gray', node_size=2, ax=ax)
        edges = [(u, v) for u, v in path[:num+1]]
        nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color='red', width=2, ax=ax)
        if num < len(path):
            u, v = path[num]
            ax.scatter([pos[u][0]], [pos[u][1]], c='green', s=100, zorder=5)
            ax.text(pos[u][0], pos[u][1], 'Start', fontsize=12, ha='center', color='green')

    ani = FuncAnimation(fig, update, frames=len(path), repeat=False)
    ani.save(filename, writer='ffmpeg', fps=80, dpi=300)

def main():
    geojson_fp = 'Data/geobase.json'
    quartier_principal = 'Outremont'
    quartiers_interet = [quartier_principal, 'Verdun', 'Anjou', 'Rivière-des-Prairies-Pointe-aux-Trembles', 'Le Plateau-Mont-Royal']

    gdf_filtered = load_and_prepare_data(geojson_fp, quartiers_interet)
    G = build_graph_from_gdf(gdf_filtered)

    gdf_principal = gdf_filtered[(gdf_filtered['ARR_GCH'] == quartier_principal) | (gdf_filtered['ARR_DRT'] == quartier_principal)]
    G_principal = build_graph_from_gdf(gdf_principal)

    largest_scc = max(nx.strongly_connected_components(G_principal), key=len)
    plot_graph_with_subgraph(G, largest_scc, f'{quartier_principal} - General and Subgraph', f'{quartier_principal}_graph.png')

    try:
        subgraph = G.subgraph(largest_scc).copy()
        cpp_path, cpp_length = find_eulerian_path(subgraph)
        print(f"CPP for {quartier_principal}: Length = {cpp_length:.2f} meters")
        animate_cpp(subgraph, cpp_path, f'{quartier_principal}_cpp.mp4')
    except nx.NetworkXError as e:
        print(f"Error in {quartier_principal}: {e}")

if __name__ == "__main__":
    main()
