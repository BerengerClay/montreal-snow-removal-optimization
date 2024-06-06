import geopandas as gpd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import timeit

def load_and_prepare_data(geojson_fp, quartier_interet):
    gdf = gpd.read_file(geojson_fp)

    if gdf.crs is None:
        gdf.set_crs(epsg=4326, inplace=True)

    gdf_filtered = gdf[(gdf['ARR_GCH'] == quartier_interet) | 
                       (gdf['ARR_DRT'] == quartier_interet)]
    
    return gdf_filtered

def build_graph_from_gdf(gdf):
    G = nx.DiGraph()

    for _, row in gdf.iterrows():
        coords = list(row.geometry.coords)
        sens_cir = row['SENS_CIR']
        for i in range(len(coords) - 1):
            u = (coords[i][1], coords[i][0])
            v = (coords[i + 1][1], coords[i + 1][0])
            weight = np.linalg.norm(np.array(u) - np.array(v))
            if sens_cir == 1:
                G.add_edge(u, v, weight=weight, direction='one-way')
            elif sens_cir == -1:
                G.add_edge(v, u, weight=weight, direction='one-way')
            else:
                G.add_edge(u, v, weight=weight, direction='two-way')
                G.add_edge(v, u, weight=weight, direction='two-way')
    
    return G

def add_required_edges_for_scc(G, scc):
    in_degrees = G.in_degree(scc)
    out_degrees = G.out_degree(scc)

    nodes_with_excess_in = []
    nodes_with_excess_out = []

    for node in scc:
        in_degree = in_degrees[node]
        out_degree = out_degrees[node]
        if in_degree > out_degree:
            nodes_with_excess_in.extend([node] * (in_degree - out_degree))
        elif out_degree > in_degree:
            nodes_with_excess_out.extend([node] * (out_degree - in_degree))

    for u, v in zip(nodes_with_excess_out, nodes_with_excess_in):
        G.add_edge(u, v, weight=0)
    
    return G

def solve_cpp_directed(G):
    # Ensure the graph is strongly connected
    if not nx.is_strongly_connected(G):
        components = list(nx.strongly_connected_components(G))
        for i in range(len(components) - 1):
            u = list(components[i])[0]
            v = list(components[i + 1])[0]
            G.add_edge(u, v, weight=0)
    
    # Add required edges within each strongly connected component
    sccs = list(nx.strongly_connected_components(G))
    for scc in sccs:
        G = add_required_edges_for_scc(G, scc)
    
    if not nx.is_eulerian(G):
        raise nx.NetworkXError("G is not Eulerian after adding required edges.")
    
    path = list(nx.eulerian_circuit(G))
    
    length = sum(G[u][v]['weight'] if 'weight' in G[u][v] else np.linalg.norm(np.array(u) - np.array(v)) for u, v in path)
    
    return path, length

def plot_graph_and_path(G, path, title, filename):
    pos = {node: (node[1], node[0]) for node in G.nodes()}
    fig, ax = plt.subplots()

    nx.draw(G, pos, with_labels=False, node_color='lightblue', edge_color='gray', node_size=5, ax=ax)
    
    if path:
        edges = [(u, v) for u, v in path]
        nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color='red', width=2, ax=ax, arrows=True)
        cmap = plt.cm.get_cmap('viridis', len(path))

        for i, (u, v) in enumerate(path):
            color = cmap(i)
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], edge_color=[color], width=2, ax=ax, arrows=True, arrowsize=10)
        
        start_node = path[0][0]
        plt.scatter([pos[start_node][0]], [pos[start_node][1]], c='green', s=100, zorder=5)
        plt.text(pos[start_node][0], pos[start_node][1], 'Start', fontsize=12, ha='center', color='green')
    
    plt.title(title)
    plt.savefig(filename, format='png', dpi=300)
    plt.close()

def main():
    geojson_fp = 'Data/geobase.json'
    quartier_interet = 'Outremont'

    gdf_filtered = load_and_prepare_data(geojson_fp, quartier_interet)
    G = build_graph_from_gdf(gdf_filtered)
    
    start_time = timeit.default_timer()
    try:
        cpp_path, cpp_length = solve_cpp_directed(G)
        end_time = timeit.default_timer()
        cpp_time = end_time - start_time

        print(f"CPP - Time: {cpp_time:.4f}s, Length: {cpp_length:.4f}")
    
        plot_graph_and_path(G, cpp_path, 'Chinese Postman Problem - Outremont', 'outremont_cpp.png')
    except nx.NetworkXError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
