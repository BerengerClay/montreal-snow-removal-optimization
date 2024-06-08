import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from matplotlib.colors import Normalize

def generate_city_graph(grid_size, edge_prob=0.5, extra_edges=5, one_way_prob=0.3, seed=43):
    np.random.seed(seed)
    G = nx.DiGraph()

    for i in range(grid_size):
        for j in range(grid_size):
            G.add_node((i, j))

    for i in range(grid_size):
        for j in range(grid_size):
            if i < grid_size - 1 and np.random.rand() < edge_prob:
                if np.random.rand() < one_way_prob:
                    G.add_edge((i, j), (i + 1, j), weight=np.linalg.norm(np.array((i,j)) - np.array((i+1,j))), direction='one-way')
                else:
                    G.add_edge((i, j), (i + 1, j), weight=np.linalg.norm(np.array((i,j)) - np.array((i+1,j))), direction='two-way')
                    G.add_edge((i + 1, j), (i, j), weight=np.linalg.norm(np.array((i+1,j)) - np.array((i,j))), direction='two-way')
            if j < grid_size - 1 and np.random.rand() < edge_prob:
                if np.random.rand() < one_way_prob:
                    G.add_edge((i, j), (i, j + 1), weight=np.linalg.norm(np.array((i,j)) - np.array((i,j + 1))), direction='one-way')
                else:
                    G.add_edge((i, j), (i, j + 1), weight=np.linalg.norm(np.array((i,j)) - np.array((i,j + 1))), direction='two-way')
                    G.add_edge((i, j + 1), (i, j), weight=np.linalg.norm(np.array((i,j + 1)) - np.array((i,j))), direction='two-way')

    for _ in range(extra_edges):
        u = (np.random.randint(0, grid_size), np.random.randint(0, grid_size))
        v = (np.random.randint(0, grid_size), np.random.randint(0, grid_size))
        if u != v and not G.has_edge(u, v):
            G.add_edge(u, v, weight=np.linalg.norm(np.array(u) - np.array(v)), direction='two-way')
            G.add_edge(v, u, weight=np.linalg.norm(np.array(v) - np.array(u)), direction='two-way')

    if not nx.is_strongly_connected(G):
        components = list(nx.strongly_connected_components(G))
        for i in range(len(components) - 1):
            u = list(components[i])[0]
            v = list(components[i + 1])[0]
            G.add_edge(u, v, weight=np.linalg.norm(np.array(u) - np.array(v)), direction='auxiliary')
            G.add_edge(v, u, weight=np.linalg.norm(np.array(v) - np.array(u)), direction='auxiliary')

    return G

def generate_clusters(G, num_clusters, seed=0):
    np.random.seed(seed)
    pos = np.array([node for node in G.nodes()])
    kmeans = KMeans(n_clusters=num_clusters, random_state=seed).fit(pos)
    clusters = kmeans.labels_
    return clusters

def plot_oriented_graph(G, clusters, snow_intensity, min_intensity, max_intensity, grid_size, ax):
    pos = {node: (node[0], node[1]) for node in G.nodes()}
    norm = Normalize(vmin=min_intensity, vmax=max_intensity)
    node_colors = [plt.cm.Blues(norm(snow_intensity[cluster])) for cluster in clusters]
    
    edge_colors = []
    for u, v in G.edges():
        u_cluster = clusters[u[0] * grid_size + u[1]]
        v_cluster = clusters[v[0] * grid_size + v[1]]
        avg_intensity = (snow_intensity[u_cluster] + snow_intensity[v_cluster]) / 2
        edge_colors.append(plt.cm.Blues(norm(avg_intensity)))

    nx.draw_networkx_edges(G, pos, edgelist=[(u, v) for u, v, d in G.edges(data=True) if d['direction'] == 'two-way'], edge_color=edge_colors, width=2.0, ax=ax, arrows=False)
    nx.draw_networkx_edges(G, pos, edgelist=[(u, v) for u, v, d in G.edges(data=True) if d['direction'] == 'one-way'], edge_color=edge_colors, width=2.0, ax=ax, arrows=True, arrowstyle='-|>', arrowsize=10)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=0, ax=ax)
    ax.set_title("City Graph with Snow Intensity")

def main():
    grid_size = 40  # Taille de la grille
    edge_prob = 0.6
    extra_edges = int(1/100 * grid_size ** 2)
    one_way_prob = 0.3
    num_clusters = 50

    # Définir les valeurs minimales et maximales de l'intensité de la neige
    min_snow_intensity = 0
    max_snow_intensity = 15

    G = generate_city_graph(grid_size, edge_prob, extra_edges, one_way_prob)
    clusters = generate_clusters(G, num_clusters)
    
    # Générer une intensité de neige pour chaque cluster dans l'intervalle défini
    snow_intensity = np.random.uniform(min_snow_intensity, max_snow_intensity, num_clusters)

    fig, ax = plt.subplots(figsize=(8, 8))
    plot_oriented_graph(G, clusters, snow_intensity, min_snow_intensity, max_snow_intensity, grid_size, ax)
    plt.show()

if __name__ == "__main__":
    main()
