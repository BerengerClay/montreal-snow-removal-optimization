import geopandas as gpd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

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

def plot_graphs(G_one_way_positive, G_one_way_negative, G_two_way, title):
    pos = {node: (node[1], node[0]) for node in set(G_one_way_positive.nodes()).union(set(G_one_way_negative.nodes())).union(set(G_two_way.nodes()))}
    fig, ax = plt.subplots()

    nx.draw(G_two_way, pos, with_labels=False, node_color='lightblue', edge_color='blue', node_size=5, ax=ax, label='Two-way streets')
    nx.draw(G_one_way_positive, pos, with_labels=False, node_color='lightblue', edge_color='red', node_size=5, ax=ax, label='One-way streets')
    nx.draw(G_one_way_negative, pos, with_labels=False, node_color='lightblue', edge_color='red', node_size=5, ax=ax, label='One-way streets')
    
    plt.title(title)
    plt.legend(handles=[
        plt.Line2D([0], [0], color='blue', lw=2, label='Two-way streets'),
        plt.Line2D([0], [0], color='red', lw=2, label='One-way streets'),
        #plt.Line2D([0], [0], color='green', lw=2, label='One-way streets (negative)')
    ])
    plt.show()

def main():
    geojson_fp = 'Data/geobase.json'
    # quartiers_interet = ['Outremont', 'Verdun', 'Anjou', 
    #                      'Rivière-des-Prairies-Pointe-aux-Trembles', 
    #                      'Le Plateau-Mont-Royal']
    
    quartiers_interet = ['Verdun']

    gdf_filtered = load_and_prepare_data(geojson_fp, quartiers_interet)
    G_one_way_positive, G_one_way_negative, G_two_way = build_graphs_from_gdf(gdf_filtered)
    
    plot_graphs(G_one_way_positive, G_one_way_negative, G_two_way, 'Montreal Streets - One-way (Red/Green) vs Two-way (Blue)')

if __name__ == "__main__":
    main()
