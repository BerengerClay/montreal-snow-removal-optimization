import geopandas as gpd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.animation import FuncAnimation
from collections import defaultdict
from networkx.algorithms.approximation import min_weighted_vertex_cover
import timeit

def load_and_prepare_data(geojson_fp, quartiers_interet):
    gdf = gpd.read_file(geojson_fp)

    if gdf.crs is None:
        gdf.set_crs(epsg=4326, inplace=True)

    gdf_filtered = gdf[(gdf['ARR_GCH'].isin(quartiers_interet)) | 
                       (gdf['ARR_DRT'].isin(quartiers_interet))]
    
    return gdf_filtered

def build_graph_from_gdf(gdf):
    G = nx.Graph()

    for _, row in gdf.iterrows():
        coords = list(row.geometry.coords)
        for i in range(len(coords) - 1):
            u = (coords[i][1], coords[i][0])
            v = (coords[i + 1][1], coords[i + 1][0])
            weight = np.linalg.norm(np.array(u) - np.array(v))
            G.add_edge(u, v, weight=weight)
    
    return G

def ensure_connected(G):
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G_connected = G.subgraph(largest_cc).copy()
        return G_connected
    return G


def split_graph(G):
    G = ensure_connected(G)
    
    if not nx.is_eulerian(G):
        G = nx.eulerize(G)

    nodes_sorted_by_lat = sorted(G.nodes(), key=lambda x: x[0])
    index = len(nodes_sorted_by_lat) // 8

    nodes1 = set(nodes_sorted_by_lat[:index])
    nodes2 = set(nodes_sorted_by_lat[index:2*index])
    nodes3 = set(nodes_sorted_by_lat[2*index:3*index])
    nodes4 = set(nodes_sorted_by_lat[3*index:4*index])
    nodes5 = set(nodes_sorted_by_lat[4*index:5*index])
    nodes6 = set(nodes_sorted_by_lat[5*index:6*index])
    nodes7 = set(nodes_sorted_by_lat[6*index:7*index])
    nodes8 = set(nodes_sorted_by_lat[7*index:8*index])

    G1 = G.subgraph(nodes1).copy()
    G2 = G.subgraph(nodes2).copy()
    G3 = G.subgraph(nodes3).copy()
    G4 = G.subgraph(nodes4).copy()
    G5 = G.subgraph(nodes5).copy()
    G6 = G.subgraph(nodes6).copy()
    G7 = G.subgraph(nodes7).copy()
    G8 = G.subgraph(nodes8).copy()
    
    L = []
    for i in range(0,8):
        nodes1 = set(nodes_sorted_by_lat[i*index:(i+1)*index])
        G1 = G.subgraph(nodes1).copy()
        L.append(G1)

    return L


def solve_cpp(G):
    G = ensure_connected(G)
    
    if nx.is_eulerian(G):
        path = list(nx.eulerian_circuit(G))
    else:
        G = nx.eulerize(G)
        path = list(nx.eulerian_circuit(G))
    
    length = sum(np.linalg.norm(np.array(u) - np.array(v)) for u, v in path)
        
    return path, length

def plot_graph_and_path(G, path, save_path='one_postman.mp4'):
    pos = {node: (node[1], node[0]) for node in G.nodes()}
    fig, ax = plt.subplots()

    def update(num):
        ax.clear()
        nx.draw(G, pos, with_labels=False, node_color='lightblue', edge_color='gray', node_size=2, ax=ax)
        edges = [(u, v) for u, v in path[:num+1]]
        nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color='red', width=2, ax=ax)

        start_node = path[0][0]

        ax.scatter([pos[start_node][0]], [pos[start_node][1]], c='green', s=50, zorder=5)

    ani = FuncAnimation(fig, update, frames=len(path), repeat=False)
    ani.save(save_path, writer='ffmpeg', fps=80, dpi=300)

def plot_graph_and_paths(G, paths, save_path='two_postmen.mp4', fps=100, dpi=300):
    pos = {node: (node[1], node[0]) for node in G.nodes()}
    fig, ax = plt.subplots()

    def update(num):
        ax.clear()
        nx.draw(G, pos, with_labels=False, node_color='lightblue', edge_color='gray', node_size=5, ax=ax)

        for e in paths:
            edges = [(u, v) for u, v in e[:num+1]]
            nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color='red', width=2, ax=ax)

        colors = itertools.cycle(plt.cm.get_cmap('tab20').colors)

        for e in paths:
            if (e):
                colors = next(colors)
                start_node = e[0][0]
                ax.scatter([pos[start_node][0]], [pos[start_node][1]], c=colors, s=50, zorder=5)

    max_length = 0
    for e in paths:
        if (len(e) > max_length):
            max_length = len(e)

    ani = FuncAnimation(fig, update, frames=max_length, repeat=False)
    ani.save(save_path, writer='ffmpeg', fps=fps, dpi=dpi, codec='libx264', extra_args=['-pix_fmt', 'yuv420p'])
    plt.show()

def main():
    geojson_fp = 'Data/geobase.json'
    quartiers_interet = ['Outremont']

    gdf_filtered = load_and_prepare_data(geojson_fp, quartiers_interet)
    G = build_graph_from_gdf(gdf_filtered)
    
    L = split_graph(G)
    
    start_time = timeit.default_timer()
    
    cpp_paths = []
    cpp_lengths = []

    for i in range(0,8):
        cpp_path1, cpp_length1 = solve_cpp(L[i])
        cpp_paths.append(cpp_path1)
        cpp_lengths.append(cpp_length1)

    end_time = timeit.default_timer()
    cpp_time = end_time - start_time
    
 #   print(f"CPP - Time: {cpp_time:.4f}s, Length 1: {cpp_length1:.4f}, Length 2: {cpp_length2:.4f}")
    
    plot_graph_and_paths(G, cpp_paths, save_path='8_postmen.mp4', fps=100, dpi=300)

    print("finito")

if __name__ == "__main__":
    main()
