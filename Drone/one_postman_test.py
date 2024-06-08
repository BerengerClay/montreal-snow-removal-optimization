import geopandas as gpd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import timeit
import matplotlib.cm as cm

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
        components = list(nx.connected_components(G))
        for i in range(len(components) - 1):
            u = list(components[i])[0]
            v = list(components[i + 1])[0]
            G.add_edge(u, v, weight=0)
    return G

def solve_cpp(G):
    G = ensure_connected(G)
    
    if nx.is_eulerian(G):
        path = list(nx.eulerian_circuit(G))
    else:
        G = nx.eulerize(G)
        path = list(nx.eulerian_circuit(G))
    
    length = sum(np.linalg.norm(np.array(u) - np.array(v)) for u, v in path)
        
    return path, length

def plot_graph_and_path(G, path, title):
    pos = {node: (node[1], node[0]) for node in G.nodes()}
    nx.draw(G, pos, with_labels=False, node_color='lightblue', edge_color='gray', node_size=5)
    
    if path:
        edges = [(u, v) for u, v in path]
        nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color='red', width=2)

        start_node = path[0][0]

        plt.scatter([pos[start_node][0]], [pos[start_node][1]], c='green', s=200, zorder=5)

    plt.title(title)
    plt.show()

def main():
    geojson_fp = 'Data/geobase.json'
    # quartiers_interet = ['Outremont', 'Verdun', 'Anjou', 
    #                      'Rivière-des-Prairies-Pointe-aux-Trembles', 
    #                      'Le Plateau-Mont-Royal']
    quartiers_interet = ['Outremont']

    gdf_filtered = load_and_prepare_data(geojson_fp, quartiers_interet)
    G = build_graph_from_gdf(gdf_filtered)
    
    start_time = timeit.default_timer()
    cpp_path, cpp_length = solve_cpp(G)
    end_time = timeit.default_timer()
    cpp_time = end_time - start_time
    
    print(f"CPP - Time: {cpp_time:.4f}s, Length: {cpp_length:.4f}")
    
    plot_graph_and_path(G, cpp_path, 'Chinese Postman Problem - Montreal')

if __name__ == "__main__":
    main()
