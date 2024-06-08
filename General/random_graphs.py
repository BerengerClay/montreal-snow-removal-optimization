import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def generate_city_graph(grid_size, edge_prob=0.5, extra_edges=5, seed=42):
    np.random.seed(seed)
    G = nx.Graph()
    coords = []

    for i in range(grid_size):
        for j in range(grid_size):
            coords.append((i, j))
            G.add_node((i, j))

    for i in range(grid_size):
        for j in range(grid_size):
            if i < grid_size - 1 and np.random.rand() < edge_prob:
                G.add_edge((i, j), (i + 1, j), weight=np.random.rand())
            if j < grid_size - 1 and np.random.rand() < edge_prob:
                G.add_edge((i, j), (i, j + 1), weight=np.random.rand())

    for _ in range(extra_edges):
        u = (np.random.randint(0, grid_size), np.random.randint(0, grid_size))
        v = (np.random.randint(0, grid_size), np.random.randint(0, grid_size))
        if u != v and not G.has_edge(u, v):
            G.add_edge(u, v, weight=np.random.rand())

    if not nx.is_connected(G):
        components = list(nx.connected_components(G))
        for i in range(len(components) - 1):
            u = list(components[i])[0]
            v = list(components[i + 1])[0]
            G.add_edge(u, v, weight=np.random.rand())
    
    return G, coords

def plot_graph(G, coords, ax, title, node_size):
    pos = {node: (node[0], node[1]) for node in G.nodes()}
    nx.draw(G, pos, with_labels=False, node_color='lightblue', edge_color='gray', node_size=node_size, ax=ax)
    ax.set_title(title)

# def main():
#     grid_sizes = range(3, 12)
#     fig, axes = plt.subplots(3, 3, figsize=(15, 15))

#     for i, grid_size in enumerate(grid_sizes):
#         ax = axes[i // 3, i % 3]
#         edge_prob = 0.6
#         extra_edges = int(1/100 * grid_size ** 2)
#         G, coords = generate_city_graph(grid_size, edge_prob, extra_edges)
        
#         node_size = 300 / grid_size
#         plot_graph(G, coords, ax, f'Grid Size: {grid_size}', node_size)

#     for j in range(i + 1, 9):
#         fig.delaxes(axes[j // 3, j % 3])

#     plt.tight_layout()
#     plt.show()

def plot_graph(G, coords, title, node_size):
    pos = {node: (node[0], node[1]) for node in G.nodes()}
    plt.figure(figsize=(8, 8))
    nx.draw(G, pos, with_labels=False, node_color='lightblue', edge_color='gray', node_size=node_size)
    plt.title(title)
    plt.show()

def main():
    grid_size = 50
    edge_prob = 0.6
    extra_edges = int(1/100 * grid_size ** 2)
    G, coords = generate_city_graph(grid_size, edge_prob, extra_edges)
    
    node_size = 300 / grid_size
    plot_graph(G, coords, f'Grid Size: {grid_size}', node_size)


if __name__ == "__main__":
    main()
