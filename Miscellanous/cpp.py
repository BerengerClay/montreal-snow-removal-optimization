import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import timeit
import matplotlib.cm as cm

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

def solve_cpp(G):
    if nx.is_eulerian(G):
        path = list(nx.eulerian_circuit(G))
    else:
        G = nx.eulerize(G)
        path = list(nx.eulerian_circuit(G))
    
    length = sum(G[u][v][0]['weight'] for u, v in path)
    return path, length

def plot_graph_and_path(G, coords, path, title):
    """Plot the graph and the path on the given axes."""
    pos = {node: (node[0], node[1]) for node in G.nodes()}
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray')
    
    if path:
        edges = [(u, v) for u, v in path]
        nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color='red', width=2)
        cmap = cm.get_cmap('viridis', len(path))

        label_offset = 0.1
        for i, (u, v) in enumerate(path):
            color = cmap(i)
            mid_x = (pos[u][0] + pos[v][0]) / 2
            mid_y = (pos[u][1] + pos[v][1]) / 2
            offset_x = (pos[v][1] - pos[u][1]) * label_offset
            offset_y = (pos[u][0] - pos[v][0]) * label_offset
            plt.text(mid_x + offset_x, mid_y + offset_y, str(i), fontsize=12, ha='center', va='center', color=color)
        
        start_node = path[0][0]
        start_offset_x = -0.1
        start_offset_y = -0.1
        plt.scatter([pos[start_node][0]], [pos[start_node][1]], c='green', s=200, zorder=5)
        plt.text(pos[start_node][0] + start_offset_x, pos[start_node][1] + start_offset_y, 'Start', fontsize=12, ha='center', color='green')
    
    plt.title(title)
    plt.show()

def main():
    grid_size = 5
    edge_prob = 0.5
    extra_edges = 5
    G, coords = generate_city_graph(grid_size, edge_prob, extra_edges)
    
    start_time = timeit.default_timer()
    cpp_path, cpp_length = solve_cpp(G)
    end_time = timeit.default_timer()
    cpp_time = end_time - start_time
    
    print(f"CPP - Time: {cpp_time:.4f}s, Length: {cpp_length:.4f}")
    
    plot_graph_and_path(G, coords, cpp_path, 'Chinese Postman Problem')

if __name__ == "__main__":
    main()
