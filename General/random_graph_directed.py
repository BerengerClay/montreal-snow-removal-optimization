import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

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

def plot_oriented_graph(G, title, node_size, ax):
    pos = {node: (node[0], node[1]) for node in G.nodes()}

    nx.draw_networkx_edges(G, pos, edgelist=[(u, v) for u, v, d in G.edges(data=True) if d['direction'] == 'two-way'], edge_color='black', width=1.0, ax=ax, arrows=False)
    
    nx.draw_networkx_edges(G, pos, edgelist=[(u, v) for u, v, d in G.edges(data=True) if d['direction'] == 'one-way'], edge_color='gray', width=1.0, ax=ax, arrows=True, arrowstyle='-|>', arrowsize=10)

    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=node_size, ax=ax)
    ax.set_title(title)

def main():
    grid_size = 4  # Taille de la grille
    edge_prob = 0.6
    extra_edges = int(1/100 * grid_size ** 2)
    one_way_prob = 0.3
    G = generate_city_graph(grid_size, edge_prob, extra_edges, one_way_prob)

    fig, ax = plt.subplots(figsize=(8, 8))
    plot_oriented_graph(G, f'Oriented Grid Size: {grid_size}', node_size=300 / grid_size, ax=ax)
    plt.show()

if __name__ == "__main__":
    main()
