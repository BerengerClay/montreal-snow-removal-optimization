import geopandas as gpd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.animation import FuncAnimation
import timeit
from geopy.distance import geodesic

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
            weight = geodesic((coords[i][1], coords[i][0]), (coords[i + 1][1], coords[i + 1][0])).meters
            G.add_edge(u, v, weight=weight)
    
    return G

def ensure_connected(G):
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G_connected = G.subgraph(largest_cc).copy()
        return G_connected
    return G

def solve_cpp(G):
    G = ensure_connected(G)
    
    if nx.is_eulerian(G):
        path = list(nx.eulerian_circuit(G))
    else:
        G = nx.eulerize(G)
        path = list(nx.eulerian_circuit(G))
    
    length = sum(G[u][v].get('weight', geodesic((u[1], u[0]), (v[1], v[0])).meters) for u, v in path)
        
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

def main():
    geojson_fp = 'Data/geobase.json'
    quartiers_interet = ['Outremont']

    gdf_filtered = load_and_prepare_data(geojson_fp, quartiers_interet)
    G = build_graph_from_gdf(gdf_filtered)
    
    start_time = timeit.default_timer()
    cpp_path, cpp_length = solve_cpp(G)
    end_time = timeit.default_timer()
    cpp_time = end_time - start_time
    
    print(f"CPP - Time: {cpp_time:.4f}s, Length: {cpp_length:.2f} meters")
    
    #plot_graph_and_path(G, cpp_path)

    print("finito")

if __name__ == "__main__":
    main()
