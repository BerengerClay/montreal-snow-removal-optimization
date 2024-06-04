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
        # components = list(nx.connected_components(G))
        # for i in range(len(components) - 1):
        #     u = list(components[i])[0]
        #     v = list(components[i + 1])[0]
        #     G.add_edge(u, v, weight=0)
        largest_cc = max(nx.connected_components(G), key=len)
        G_connected = G.subgraph(largest_cc).copy()
        return G_connected
    return G


def split_graph(G):
    G = ensure_connected(G)
    
    if not nx.is_eulerian(G):
        G = nx.eulerize(G)

    # Sort nodes by latitude (y-coordinate)
    nodes_sorted_by_lat = sorted(G.nodes(), key=lambda x: x[0])
    mid_index = len(nodes_sorted_by_lat) // 2

    # Split nodes into two groups
    nodes1 = set(nodes_sorted_by_lat[:mid_index+1])
    nodes2 = set(nodes_sorted_by_lat[mid_index-1:])

    G1 = G.subgraph(nodes1).copy()
    G2 = G.subgraph(nodes2).copy()
    
    return G1, G2


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
    #plt.show()

def plot_graph_and_paths(G, path1, path2, save_path='two_postmen.mp4', fps=100, dpi=300):
    pos = {node: (node[1], node[0]) for node in G.nodes()}
    fig, ax = plt.subplots()
    edge_counts = defaultdict(int)
    cmap1 = cm.get_cmap('cool', max(len(path1), 10))  # Colormap for postman 1
    cmap2 = cm.get_cmap('hot', max(len(path2), 10))   # Colormap for postman 2

    def update(num):
        ax.clear()
        nx.draw(G, pos, with_labels=False, node_color='lightblue', edge_color='gray', node_size=5, ax=ax)

        edges1 = [(u, v) for u, v in path1[:num+1]]
        nx.draw_networkx_edges(G, pos, edgelist=edges1, edge_color='red', width=2, ax=ax)

        edges2 = [(u, v) for u, v in path2[:num+1]]
        nx.draw_networkx_edges(G, pos, edgelist=edges2, edge_color='blue', width=2, ax=ax)

        # for i in range(num + 1):
        #     if i < len(path1):
        #         u1, v1 = path1[i]
        #         edge_counts[(u1, v1)] += 1
        #         edge_counts[(v1, u1)] += 1
        #         color1 = cmap1(edge_counts[(u1, v1)])
        #         nx.draw_networkx_edges(G, pos, edgelist=[(u1, v1)], edge_color=[color1], width=2, ax=ax)
                
        #     if i < len(path2):
        #         u2, v2 = path2[i]
        #         edge_counts[(u2, v2)] += 1
        #         edge_counts[(v2, u2)] += 1
        #         color2 = cmap2(edge_counts[(u2, v2)])
        #         nx.draw_networkx_edges(G, pos, edgelist=[(u2, v2)], edge_color=[color2], width=2, ax=ax)

        if path1:
            start_node1 = path1[0][0]
            ax.scatter([pos[start_node1][0]], [pos[start_node1][1]], c='green', s=50, zorder=5)
        
        if path2:
            start_node2 = path2[0][0]
            ax.scatter([pos[start_node2][0]], [pos[start_node2][1]], c='red', s=50, zorder=5)

    max_length = max(len(path1), len(path2))
    ani = FuncAnimation(fig, update, frames=max_length, repeat=False)
    ani.save(save_path, writer='ffmpeg', fps=fps, dpi=dpi, codec='libx264', extra_args=['-pix_fmt', 'yuv420p'])
    plt.show()

def main():
    geojson_fp = 'Data/geobase.json'
    quartiers_interet = ['Outremont']

    gdf_filtered = load_and_prepare_data(geojson_fp, quartiers_interet)
    G = build_graph_from_gdf(gdf_filtered)
    
    G1, G2 = split_graph(G)
    
    start_time = timeit.default_timer()
    cpp_path1, cpp_length1 = solve_cpp(G1)
    cpp_path2, cpp_length2 = solve_cpp(G2)
    end_time = timeit.default_timer()
    cpp_time = end_time - start_time
    
    print(f"CPP - Time: {cpp_time:.4f}s, Length 1: {cpp_length1:.4f}, Length 2: {cpp_length2:.4f}")
    
    plot_graph_and_paths(G, cpp_path1, cpp_path2, save_path='two_postmen.mp4', fps=100, dpi=300)

    print("finito")

if __name__ == "__main__":
    main()
