import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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

def plot_graph(G, title, node_size, ax):
    pos = {node: (node[0], node[1]) for node in G.nodes()}
    nx.draw(G, pos, with_labels=False, node_color='lightblue', edge_color='gray', node_size=node_size, ax=ax)
    ax.set_title(title)

def make_eulerian(G):
    G_eulerian = G.copy()

    in_degrees = dict(G_eulerian.in_degree())
    out_degrees = dict(G_eulerian.out_degree())
    imbalanced_nodes = {node: out_degrees[node] - in_degrees[node] for node in G_eulerian.nodes()}

    positive_imbalance = {node: imbalance for node, imbalance in imbalanced_nodes.items() if imbalance > 0}
    negative_imbalance = {node: -imbalance for node, imbalance in imbalanced_nodes.items() if imbalance < 0}
    
    added_edges = []
    converted_edges = []

    for node, imbalance in positive_imbalance.items():
        while imbalance > 0:
            closest_node = min(negative_imbalance.keys(), key=lambda n: np.linalg.norm(np.array(node) - np.array(n)))
            if not G_eulerian.has_edge(closest_node, node):
                G_eulerian.add_edge(closest_node, node, weight=1, direction='auxiliary')
                added_edges.append((closest_node, node))
            else:
                if G_eulerian[closest_node][node]['direction'] == 'one-way':
                    G_eulerian[closest_node][node]['direction'] = 'two-way'
                else:
                    G_eulerian.add_edge(node, closest_node, weight=1, direction='auxiliary')
                converted_edges.append((closest_node, node))
            negative_imbalance[closest_node] -= 1
            if negative_imbalance[closest_node] == 0:
                del negative_imbalance[closest_node]
            imbalance -= 1

    return G_eulerian, added_edges, converted_edges

def chinese_postman_path(G):
    if not nx.is_eulerian(G):
        raise nx.NetworkXError("G is not Eulerian.")
    return list(nx.eulerian_circuit(G))

def animate_postman_path(G, path, added_edges, converted_edges, filename):
    pos = {node: (node[0], node[1]) for node in G.nodes()}
    fig, ax = plt.subplots(figsize=(8, 8))

    def update(num):
        ax.clear()
        nx.draw(G, pos, with_labels=False, node_color='lightblue', edge_color='gray', node_size=100, ax=ax)

        nx.draw_networkx_edges(G, pos, edgelist=added_edges, edge_color='green', width=2.0, ax=ax)

        nx.draw_networkx_edges(G, pos, edgelist=converted_edges, edge_color='blue', width=2.0, ax=ax)

        nx.draw_networkx_edges(G, pos, edgelist=path[:num+1], edge_color='red', width=2.0, ax=ax)

        plt.title(f'Step {num+1}')

    ani = animation.FuncAnimation(fig, update, frames=len(path), repeat=False)
    ani.save(filename, writer='ffmpeg', fps=2)

def main():
    grid_size = 4  # Taille de la grille
    edge_prob = 0.6
    extra_edges = int(1/100 * grid_size ** 2)
    G = generate_city_graph(grid_size, edge_prob, extra_edges)

    fig, ax = plt.subplots(figsize=(8, 8))
    plot_graph(G, f'Original Grid Size: {grid_size}', node_size=300 / grid_size, ax=ax)
    plt.show()

    G_eulerian, added_edges, converted_edges = make_eulerian(G)
    if nx.is_eulerian(G_eulerian):
        print("Graph is now Eulerian: True")
        path = chinese_postman_path(G_eulerian)
        animate_postman_path(G, path, added_edges, converted_edges, 'postman_animation.mp4')
    else:
        print("Graph is now Eulerian: False")

if __name__ == "__main__":
    main()
