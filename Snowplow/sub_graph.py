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

def plot_graph_with_subgraph(G, subgraph_nodes, title, filename):
    pos = {node: (node[1], node[0]) for node in G.nodes()}
    fig, ax = plt.subplots(figsize=(10, 8))

    for u, v, d in G.edges(data=True):
        if d['direction'] != 'two-way':
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], edge_color='gray', width=1, ax=ax, arrows=True,
                                   arrowstyle='-|>', arrowsize=10)

    for u, v, d in G.edges(data=True):
        if d['direction'] == 'two-way' and (v, u) in G.edges:
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], edge_color='gray', width=1, ax=ax, arrows=False)

    subgraph = G.subgraph(subgraph_nodes)
    for u, v, d in subgraph.edges(data=True):
        if d['direction'] == 'two-way' and (v, u) in subgraph.edges:
            nx.draw_networkx_edges(subgraph, pos, edgelist=[(u, v)], edge_color='green', width=1.5, ax=ax, arrows=False)
        else:
            nx.draw_networkx_edges(subgraph, pos, edgelist=[(u, v)], edge_color='green', width=1.5, ax=ax, arrows=True,
                                   arrowstyle='-|>', arrowsize=10)

    plt.title(title)
    plt.savefig(filename, format='png', dpi=300)
    plt.close()

def main():
    geojson_fp = 'Data/geobase.json'
    quartiers_interet = ['Outremont']#, 'Verdun', 'Anjou', 'Rivière-des-Prairies-Pointe-aux-Trembles', 'Le Plateau-Mont-Royal']

    gdf_filtered = load_and_prepare_data(geojson_fp, quartiers_interet)
    
    for quartier in quartiers_interet:
        gdf_quartier = gdf_filtered[(gdf_filtered['ARR_GCH'] == quartier) | (gdf_filtered['ARR_DRT'] == quartier)]
        G = build_graph_from_gdf(gdf_quartier)
        
        largest_scc = max(nx.strongly_connected_components(G), key=len)
        
        plot_graph_with_subgraph(G, largest_scc, f'{quartier} - General and Subgraph', f'{quartier}_graph.png')

if __name__ == "__main__":
    main()
