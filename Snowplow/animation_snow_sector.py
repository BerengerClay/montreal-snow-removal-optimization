import geopandas as gpd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from geopy.distance import geodesic
from sklearn.cluster import KMeans

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
                G.add_edge(u, v, weight=weight, direction='one-way', street_name=street_name)
            elif direction == -1:
                G.add_edge(v, u, weight=weight, direction='one-way', street_name=street_name)
            else:
                G.add_edge(u, v, weight=weight, direction='two-way', street_name=street_name)
                G.add_edge(v, u, weight=weight, direction='two-way', street_name=street_name)
    return G

def generate_clusters(G, num_clusters, seed=0):
    np.random.seed(seed)
    pos = np.array([node for node in G.nodes()])
    kmeans = KMeans(n_clusters=num_clusters, random_state=seed).fit(pos)
    clusters = kmeans.labels_
    return clusters

def filter_graph_by_snow_intensity(G, clusters, snow_intensity, threshold):
    nodes_to_keep = [node for node, cluster in zip(G.nodes(), clusters) if snow_intensity[cluster] > threshold]
    subgraph = G.subgraph(nodes_to_keep).copy()
    return subgraph

def plot_graph(G, title, node_size, ax):
    pos = {node: (node[1], node[0]) for node in G.nodes()}

    nx.draw_networkx_edges(G, pos, edgelist=[(u, v) for u, v, d in G.edges(data=True) if d['direction'] == 'two-way'], edge_color='black', width=1.0, ax=ax, arrows=False)
    
    nx.draw_networkx_edges(G, pos, edgelist=[(u, v) for u, v, d in G.edges(data=True) if d['direction'] == 'one-way'], edge_color='gray', width=1.0, ax=ax, arrows=True, arrowstyle='-|>', arrowsize=10)

    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=node_size, ax=ax)
    ax.set_title(title)

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
            
            if True:#path_exists:
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

def chinese_postman_path(G):
    if not nx.is_eulerian(G):
        raise nx.NetworkXError("G is not Eulerian.")
    return list(nx.eulerian_circuit(G))

def animate_postman_path(G, G_eulerian, path, added_edges, converted_edges, G_full, filename):
    pos = {node: (node[1], node[0]) for node in G_full.nodes()}
    fig, ax = plt.subplots(figsize=(8, 8))

    distances = [G_eulerian[u][v]['weight'] for u, v in path]
    cumulative_distances = np.cumsum(distances)

    def update(num):
        ax.clear()
        nx.draw_networkx_nodes(G_full, pos, node_color='lightgray', node_size=1, ax=ax)
        nx.draw_networkx_edges(G_full, pos, edge_color='lightgray', width=0.5, ax=ax)

        nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=1, ax=ax)
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v) for u, v, d in G.edges(data=True) if d['direction'] == 'two-way'], edge_color='black', width=1.0, ax=ax, arrows=False)
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v) for u, v, d in G.edges(data=True) if d['direction'] == 'one-way'], edge_color='gray', width=1.0, ax=ax, arrows=True, arrowstyle='-|>', arrowsize=10)

        nx.draw_networkx_edges(G, pos, edgelist=added_edges, edge_color='green', width=2.0, ax=ax, arrows=True, arrowstyle='-|>', arrowsize=10)
        
        nx.draw_networkx_edges(G, pos, edgelist=converted_edges, edge_color='blue', width=2.0, ax=ax, arrows=True, arrowstyle='-|>', arrowsize=10)

        nx.draw_networkx_edges(G, pos, edgelist=path[:num+1], edge_color='red', width=2.0, ax=ax, arrows=True, arrowstyle='-|>', arrowsize=10)

        current_distance = cumulative_distances[num]
        plt.title(f'Distance: {current_distance:.2f} meters')

    ani = animation.FuncAnimation(fig, update, frames=len(path), repeat=False)
    ani.save(filename, writer='ffmpeg', fps=80)

def main():
    geojson_fp = 'Data/geobase.json'
    quartiers_interet = ['Outremont']

    gdf_filtered = load_and_prepare_data(geojson_fp, quartiers_interet)
    G_full = build_graph_from_gdf(gdf_filtered)

    num_clusters = 10
    min_snow_intensity = 0
    max_snow_intensity = 20
    clusters = generate_clusters(G_full, num_clusters)
    snow_intensity = np.random.uniform(min_snow_intensity, max_snow_intensity, num_clusters)
    threshold = 13

    G_filtered = filter_graph_by_snow_intensity(G_full, clusters, snow_intensity, threshold)

    G_eulerian, added_edges, converted_edges = make_eulerian(G_filtered)

    largest_scc = max(nx.strongly_connected_components(G_eulerian), key=len)
    G_scc = G_eulerian.subgraph(largest_scc).copy()

    fig, ax = plt.subplots(figsize=(8, 8))
    plot_graph(G_scc, 'Largest Strongly Connected Component of Filtered and Eulerian Graph', node_size=1, ax=ax)
    plt.show()

    if nx.is_eulerian(G_scc):
        print("Graph is now Eulerian: True")
        path = chinese_postman_path(G_scc)
        animate_postman_path(G_scc, G_eulerian, path, added_edges, converted_edges, G_full, 'outremont_postman_animation.mp4')
    else:
        print("Graph is now Eulerian: False")

if __name__ == "__main__":
    main()
