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
    G = nx.DiGraph()

    for _, row in gdf.iterrows():
        coords = list(row.geometry.coords)
        street_name = row['NOM_VOIE'] if row['NOM_VOIE'] else 'Unknown Street'
        for i in range(len(coords) - 1):
            u = (coords[i][1], coords[i][0])
            v = (coords[i + 1][1], coords[i + 1][0])
            weight = geodesic((coords[i][1], coords[i][0]), (coords[i + 1][1], coords[i + 1][0])).meters
            direction = row['SENS_CIR']
            if direction == 1:
                G.add_edge(u, v, weight=weight, street_name=street_name, direction='one-way')
            elif direction == -1:
                G.add_edge(v, u, weight=weight, street_name=street_name, direction='one-way')
            else:
                G.add_edge(u, v, weight=weight, street_name=street_name, direction='two-way')
                G.add_edge(v, u, weight=weight, street_name=street_name, direction='two-way')
    
    return G

def ensure_connected(G):
    if not nx.is_strongly_connected(G):
        largest_scc = max(nx.strongly_connected_components(G), key=len)
        G_connected = G.subgraph(largest_scc).copy()
        return G_connected
    return G

def make_eulerian(G):
    G_eulerian = G.copy()

    in_degrees = dict(G_eulerian.in_degree())
    out_degrees = dict(G_eulerian.out_degree())
    imbalanced_nodes = {node: out_degrees[node] - in_degrees[node] for node in G_eulerian.nodes()}

    positive_imbalance = {node: imbalance for node, imbalance in imbalanced_nodes.items() if imbalance > 0}
    negative_imbalance = {node: -imbalance for node, imbalance in imbalanced_nodes.items() if imbalance < 0}
    
    added_edges = []
    converted_edges = []

    for node, imbalance in positive_imbalance.items():
        while imbalance > 0:
            if not negative_imbalance:
                print(f"No valid path found for node {node} with imbalance {imbalance}.")
                break
            closest_node = min(negative_imbalance.keys(), key=lambda n: np.linalg.norm(np.array(node) - np.array(n)))
            try:
                path_length = nx.shortest_path_length(G_eulerian, source=closest_node, target=node, weight='weight')
                path_exists = True
            except nx.NetworkXNoPath:
                path_exists = False
            
            if path_exists:
                if not G_eulerian.has_edge(closest_node, node):
                    G_eulerian.add_edge(closest_node, node, weight=1, direction='auxiliary')
                    added_edges.append((closest_node, node))
                else:
                    if G_eulerian[closest_node][node]['direction'] == 'one-way':
                        G_eulerian[closest_node][node]['direction'] = 'two-way'
                    else:
                        G_eulerian.add_edge(node, closest_node, weight=1, direction='auxiliary')
                    converted_edges.append((closest_node, node))
                negative_imbalance[closest_node] -= 1
                if negative_imbalance[closest_node] == 0:
                    del negative_imbalance[closest_node]
                imbalance -= 1
            else:
                print(f"No path exists between {closest_node} and {node}. Removing {closest_node} from consideration.")
                negative_imbalance.pop(closest_node)

    return G_eulerian, added_edges, converted_edges

def solve_cpp(G, start_node, output_file='street_lengths.txt'):
    G = ensure_connected(G)
    
    G_eulerian, added_edges, converted_edges = make_eulerian(G)

    if not nx.is_eulerian(G_eulerian):
        raise nx.NetworkXError("Graph is not Eulerian after adding edges.")

    path = list(nx.eulerian_circuit(G_eulerian, source=start_node))

    length = 0
    with open(output_file, 'w') as f:
        for u, v in path:
            edge_data = G_eulerian.get_edge_data(u, v)
            segment_length = edge_data.get('weight', geodesic(u, v).meters)
            length += segment_length
            street_name = edge_data.get('street_name', 'Unknown Street')
            f.write(f"Segment of {street_name} from {u} to {v}: {segment_length:.2f} meters\n")

    return path, length, G_eulerian, added_edges, converted_edges

def plot_graph_and_path(G, path, total_length, G_eulerian, added_edges, converted_edges, save_path='one_postman_length_diff_start.mp4'):
    pos = {node: (node[1], node[0]) for node in G.nodes()}
    fig, ax = plt.subplots()

    distance_covered = 0

    def update(num):
        nonlocal distance_covered
        ax.clear()
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=2, ax=ax)

        nx.draw_networkx_edges(G, pos, edgelist=[(u, v) for u, v, d in G.edges(data=True) if d['direction'] == 'two-way'], edge_color='black', width=1.0, ax=ax, arrows=False)
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v) for u, v, d in G.edges(data=True) if d['direction'] == 'one-way'], edge_color='gray', width=1.0, ax=ax, arrows=True, arrowstyle='-|>', arrowsize=10)

        nx.draw_networkx_edges(G, pos, edgelist=added_edges, edge_color='green', width=2.0, ax=ax, arrows=True, arrowstyle='-|>', arrowsize=10)

        nx.draw_networkx_edges(G, pos, edgelist=converted_edges, edge_color='blue', width=2.0, ax=ax, arrows=True, arrowstyle='-|>', arrowsize=10)

        edges = [(u, v) for u, v in path[:num+1]]
        nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color='red', width=2, ax=ax)

        if num > 0:
            u, v = path[num-1]
            distance_covered += G_eulerian[u][v].get('weight', geodesic(u, v).meters)
        
        ax.text(0.5, 0.95, f'Distance: {distance_covered:.2f} meters', transform=ax.transAxes, fontsize=12, verticalalignment='top', horizontalalignment='center')

    ani = FuncAnimation(fig, update, frames=len(path), repeat=False)
    ani.save(save_path, writer='ffmpeg', fps=80, dpi=300)

def main():
    geojson_fp = 'Data/geobase.json'
    quartiers_interet = ['Outremont']

    gdf_filtered = load_and_prepare_data(geojson_fp, quartiers_interet)
    G = build_graph_from_gdf(gdf_filtered)

    start_nodes = list(G.nodes())
    
    num_start_nodes = int(input("Enter the number of starting nodes: "))
    step = max(len(start_nodes) // num_start_nodes, 1)
    start_nodes = start_nodes[::step][:num_start_nodes]

    for i, start_node in enumerate(start_nodes):
        output_file = f'street_lengths_start_{i}.txt'
        animation_file = f'one_postman_length_start_{i}.mp4'

        start_time = timeit.default_timer()
        cpp_path, cpp_length, G_eulerian, added_edges, converted_edges = solve_cpp(G, start_node, output_file=output_file)
        end_time = timeit.default_timer()
        cpp_time = end_time - start_time
        
        print(f"CPP - Time: {cpp_time:.4f}s, Length: {cpp_length:.2f} meters, Start Node: {start_node}")
        
        plot_graph_and_path(G, cpp_path, cpp_length, G_eulerian, added_edges, converted_edges, save_path=animation_file)
        return

if __name__ == "__main__":
    main()
