import geopandas as gpd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from matplotlib.colors import Normalize

def load_and_prepare_data(geojson_fp, quartiers_interet):
    gdf = gpd.read_file(geojson_fp)

    if gdf.crs is None:
        gdf.set_crs(epsg=4326, inplace=True)

    gdf_filtered = gdf[(gdf['ARR_GCH'].isin(quartiers_interet)) | 
                       (gdf['ARR_DRT'].isin(quartiers_interet))]
    
    return gdf_filtered

def build_graphs_from_gdf(gdf):
    G_one_way_positive = nx.DiGraph()
    G_one_way_negative = nx.DiGraph()
    G_two_way = nx.Graph()

    for _, row in gdf.iterrows():
        coords = list(row.geometry.coords)
        sens_cir = row['SENS_CIR']
        for i in range(len(coords) - 1):
            u = (coords[i][1], coords[i][0])
            v = (coords[i + 1][1], coords[i + 1][0])
            weight = np.linalg.norm(np.array(u) - np.array(v))
            if sens_cir == 1:
                G_one_way_positive.add_edge(u, v, weight=weight)
            elif sens_cir == -1:
                G_one_way_negative.add_edge(v, u, weight=weight)
            else:
                G_two_way.add_edge(u, v, weight=weight)
    
    return G_one_way_positive, G_one_way_negative, G_two_way

def generate_clusters(G, num_clusters, seed=0):
    np.random.seed(seed)
    pos = np.array([node for node in G.nodes()])
    kmeans = KMeans(n_clusters=num_clusters, random_state=seed).fit(pos)
    clusters = kmeans.labels_
    return clusters

def plot_graph_with_snow(G, clusters, snow_intensity, min_intensity, max_intensity, title):
    pos = {node: (node[1], node[0]) for node in G.nodes()}
    norm = Normalize(vmin=min_intensity, vmax=max_intensity)
    node_colors = [plt.cm.Blues(norm(snow_intensity[cluster])) for cluster in clusters]
    
    edge_colors = []
    for u, v in G.edges():
        u_cluster = clusters[list(G.nodes()).index(u)]
        v_cluster = clusters[list(G.nodes()).index(v)]
        avg_intensity = (snow_intensity[u_cluster] + snow_intensity[v_cluster]) / 2
        edge_colors.append(plt.cm.Blues(norm(avg_intensity)))

    fig, ax = plt.subplots(figsize=(12, 12))
    nx.draw(G, pos, node_color=node_colors, edge_color=edge_colors, node_size=50, width=2, ax=ax)
    ax.set_title(title)
    plt.show()

def main():
    geojson_fp = 'Data/geobase.json'
    quartiers_interet = ['Outremont']

    gdf_filtered = load_and_prepare_data(geojson_fp, quartiers_interet)
    G_one_way_positive, G_one_way_negative, G_two_way = build_graphs_from_gdf(gdf_filtered)
    
    # Convert all graphs to undirected graphs and combine them
    G_one_way_positive_undirected = G_one_way_positive.to_undirected()
    G_one_way_negative_undirected = G_one_way_negative.to_undirected()
    G_combined = nx.compose_all([G_one_way_positive_undirected, G_one_way_negative_undirected, G_two_way])

    num_clusters = 10
    min_snow_intensity = 0
    max_snow_intensity = 5

    clusters = generate_clusters(G_combined, num_clusters)
    snow_intensity = np.random.uniform(min_snow_intensity, max_snow_intensity, num_clusters)

    plot_graph_with_snow(G_combined, clusters, snow_intensity, min_snow_intensity, max_snow_intensity, 'Outremont Streets with Snow Intensity')

if __name__ == "__main__":
    main()
