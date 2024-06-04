import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import timeit
import matplotlib.cm as cm
from tqdm import tqdm


def generate_city_graph(grid_size, edge_prob=0.5, extra_edges=5, seed=42):
    np.random.seed(seed)
    G = nx.Graph()

    for i in range(grid_size):
        for j in range(grid_size):
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

    return G

def solve_cpp(G):
    if nx.is_eulerian(G):
        path = list(nx.eulerian_circuit(G))
    else:
        G = nx.eulerize(G)
        path = list(nx.eulerian_circuit(G))
    
    length = sum(G[u][v][0]['weight'] for u, v in path)
    return path, length

def measure_time(G):
    return timeit.timeit(lambda: solve_cpp(G), number=1)

def main():
    grid_sizes = [int(i**1.4) for i in range(2,15)]
    edge_probs = [0.6]

    results = []
    edge_node_ratios = []

    for grid_size in tqdm(grid_sizes):
        for edge_prob in edge_probs:
            extra_edges = int(1/100 * grid_size ** 2)
            G = generate_city_graph(grid_size, edge_prob, extra_edges)
            time = measure_time(G)
            edge_count = len(G.edges())
            node_count = len(G.nodes())
            edge_node_ratio = edge_count / node_count
            results.append((grid_size, edge_count, time))
            edge_node_ratios.append(edge_node_ratio)

    edge_counts = np.array([r[1] for r in results]).reshape(-1, 1)
    times = np.array([r[2] for r in results])

    plt.plot(edge_counts, times, color='red', label='TSP')

    plt.xlabel('Number of Edges')
    plt.ylabel('Time (seconds)')
    plt.title('Comparison of TSP and Christofides Algorithm Times')

    average_ratio = np.mean(edge_node_ratios)
    plt.annotate(f'Average Edge/Node Ratio: {average_ratio:.2f}', xy=(0.5, 0.95), xycoords='axes fraction', ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.6))

    plt.legend()
    plt.grid(True)
    plt.show()

    print("Edge/Node Ratios:", edge_node_ratios)
    return edge_node_ratios


if __name__ == "__main__":
    main()
