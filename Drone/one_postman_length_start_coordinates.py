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
        street_name = row['NOM_VOIE'] if row['NOM_VOIE'] else 'Unknown Street'
        for i in range(len(coords) - 1):
            u = (coords[i][1], coords[i][0])
            v = (coords[i + 1][1], coords[i + 1][0])
            weight = geodesic((coords[i][1], coords[i][0]), (coords[i + 1][1], coords[i + 1][0])).meters
            G.add_edge(u, v, weight=weight, street_name=street_name)
    
    return G

def ensure_connected(G):
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G_connected = G.subgraph(largest_cc).copy()
        return G_connected
    return G

def get_closest_node(G, coord):
    closest_node = None
    min_distance = float('inf')
    for node in G.nodes():
        distance = geodesic(coord, node).meters
        if distance < min_distance:
            min_distance = distance
            closest_node = node
    return closest_node

def solve_cpp(G, start_coord, output_file='street_lengths.txt'):
    G = ensure_connected(G)
    
    if not nx.is_eulerian(G):
        G = nx.eulerize(G)

    start_node = get_closest_node(G, start_coord)
    path = list(nx.eulerian_circuit(G, source=start_node))

    length = 0
    with open(output_file, 'w') as f:
        for u, v in path:
            edge_data = G.get_edge_data(u, v)
            segment_length = edge_data.get('weight', geodesic(u, v).meters)
            length += segment_length
            street_name = edge_data.get('street_name', 'Unknown Street')
            f.write(f"Segment of {street_name} from {u} to {v}: {segment_length:.2f} meters\n")

    return path, length

def plot_graph_and_path(G, path, total_length, save_path='one_postman_length_diff_start_pos.mp4'):
    pos = {node: (node[1], node[0]) for node in G.nodes()}
    fig, ax = plt.subplots()

    distance_covered = 0

    def update(num):
        nonlocal distance_covered
        ax.clear()
        nx.draw(G, pos, with_labels=False, node_color='lightblue', edge_color='gray', node_size=2, ax=ax)
        edges = [(u, v) for u, v in path[:num+1]]
        nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color='red', width=2, ax=ax)

        start_node = path[0][0]
        ax.scatter([pos[start_node][0]], [pos[start_node][1]], c='green', s=50, zorder=5)

        if num > 0:
            u, v = path[num-1]
            distance_covered += G[u][v].get('weight', geodesic(u, v).meters)
        
        ax.text(0.5, 0.95, f'Distance: {distance_covered:.2f} meters', transform=ax.transAxes, fontsize=12, verticalalignment='top', horizontalalignment='center')

    ani = FuncAnimation(fig, update, frames=len(path), repeat=False)
    ani.save(save_path, writer='ffmpeg', fps=80, dpi=300)

def main():
    geojson_fp = 'Data/geobase.json'
    quartiers_interet = ['Outremont']

    gdf_filtered = load_and_prepare_data(geojson_fp, quartiers_interet)
    G = build_graph_from_gdf(gdf_filtered)

    start_coords = [
        ( 45.504951871290494, -73.61868615198286 )
        # (45.5208, -73.6156),
        # (45.5218, -73.6166),
        # (45.5228, -73.6176),
        # (45.5238, -73.6186),
        # (45.5248, -73.6196)
    ]

    for i, start_coord in enumerate(start_coords):
        output_file = f'street_lengths_start_pos_{i}.txt'
        animation_file = f'one_postman_length_start_pos_{i}.mp4'

        start_time = timeit.default_timer()
        cpp_path, cpp_length = solve_cpp(G, start_coord, output_file=output_file)
        end_time = timeit.default_timer()
        cpp_time = end_time - start_time
        
        print(f"CPP - Time: {cpp_time:.4f}s, Length: {cpp_length:.2f} meters, Start Coord: {start_coord}")
        
        plot_graph_and_path(G, cpp_path, cpp_length, save_path=animation_file)

if __name__ == "__main__":
    main()
