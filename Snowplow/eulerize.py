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
        street_name = row['NOM_VOIE'] if row['NOM_VOIE'] else 'Unknown Street'
        for i in range(len(coords) - 1):
            u = (coords[i][1], coords[i][0])
            v = (coords[i + 1][1], coords[i + 1][0])
            weight = np.linalg.norm(np.array(u) - np.array(v))
            if sens_cir == 1:
                G.add_edge(u, v, weight=weight, direction='one-way', street_name=street_name)
            elif sens_cir == -1:
                G.add_edge(v, u, weight=weight, direction='one-way', street_name=street_name)
            else:
                G.add_edge(u, v, weight=weight, direction='two-way', street_name=street_name)
                G.add_edge(v, u, weight=weight, direction='two-way', street_name=street_name)
    return G

def add_edges_to_make_eulerian(G, odd_nodes):
    """Add edges between odd degree nodes to make the graph Eulerian."""
    while len(odd_nodes) > 1:
        u = odd_nodes.pop()
        min_distance = float('inf')
        closest_node = None
        for v in odd_nodes:
            distance = np.linalg.norm(np.array(u) - np.array(v))
            if distance < min_distance:
                min_distance = distance
                closest_node = v
        odd_nodes.remove(closest_node)
        G.add_edge(u, closest_node, weight=min_distance, direction='auxiliary')
        G.add_edge(closest_node, u, weight=min_distance, direction='auxiliary')

def plot_graphs(G_original, G_eulerian, title, filename):
    pos = {node: (node[1], node[0]) for node in G_original.nodes()}
    pos.update({node: (node[1], node[0]) for node in G_eulerian.nodes()})
    fig, ax = plt.subplots(figsize=(10, 8))

    for u, v, d in G_original.edges(data=True):
        color = 'lightgray'
        arrows = True if d['direction'] != 'two-way' else False
        nx.draw_networkx_edges(G_original, pos, edgelist=[(u, v)], edge_color=color, width=1, ax=ax, arrows=arrows, arrowstyle='-|>', arrowsize=10)

    for u, v, d in G_eulerian.edges(data=True):
        if d['direction'] == 'auxiliary':
            color = 'red'
        elif d['direction'] == 'two-way' and (v, u) in G_eulerian.edges:
            color = 'green'
        else:
            color = 'blue'
        arrows = True if d['direction'] != 'two-way' else False
        nx.draw_networkx_edges(G_eulerian, pos, edgelist=[(u, v)], edge_color=color, width=1.5, ax=ax, arrows=arrows, arrowstyle='-|>', arrowsize=10)

    plt.title(title)
    plt.savefig(filename, format='png', dpi=300)
    plt.close()

def main():
    geojson_fp = 'Data/geobase.json'
    quartiers_interet = ['Outremont']#, 'Verdun', 'Anjou', 'Rivière-des-Prairies-Pointe-aux-Trembles', 'Le Plateau-Mont-Royal']
    quartier_principal = 'Outremont'

    gdf_filtered = load_and_prepare_data(geojson_fp, quartiers_interet)
    G_principal = build_graph_from_gdf(gdf_filtered[gdf_filtered['ARR_GCH'] == quartier_principal])
    G_voisin = build_graph_from_gdf(gdf_filtered[gdf_filtered['ARR_GCH'] != quartier_principal])
    
    G_combined = nx.compose(G_principal, G_voisin)
    
    odd_degree_nodes = [node for node, degree in G_principal.degree() if degree % 2 != 0]

    add_edges_to_make_eulerian(G_combined, odd_degree_nodes)
    
    is_eulerian = nx.is_eulerian(G_combined)
    print(f"Graph is now Eulerian: {is_eulerian}")

    plot_graphs(G_principal, G_combined, f'{quartier_principal} - Original and Eulerian Graph', f'{quartier_principal}_graph.png')

if __name__ == "__main__":
    main()
