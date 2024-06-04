import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import timeit
from networkx.algorithms.approximation import traveling_salesman_problem, christofides

def generate_city_graph(num_points, radius, seed=42):
    np.random.seed(seed)
    G = nx.random_geometric_graph(num_points, radius)
    pos = nx.get_node_attributes(G, 'pos')
    
    for u, v in G.edges():
        G[u][v]['weight'] = np.linalg.norm(np.array(pos[u]) - np.array(pos[v]))
    
    return G, pos

def make_complete_graph(G, pos):
    complete_G = nx.Graph()
    for u in G.nodes:
        for v in G.nodes:
            if u != v:
                weight = np.linalg.norm(np.array(pos[u]) - np.array(pos[v]))
                complete_G.add_edge(u, v, weight=weight)
    return complete_G

def measure_time(algorithm, G, pos):
    if algorithm == 'tsp':
        complete_G = make_complete_graph(G, pos)
        time_taken = timeit.timeit(lambda: traveling_salesman_problem(complete_G, cycle=True), number=1)
    elif algorithm == 'christofides':
        complete_G = make_complete_graph(G, pos)
        time_taken = timeit.timeit(lambda: christofides(complete_G), number=1)
    return time_taken

def plot_graph_and_path(G, pos, path, title):
    plt.figure(figsize=(8, 8))
    nx.draw(G, pos, node_size=50, node_color='blue', with_labels=False, edge_color='gray')
    
    if path:
        edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
        nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color='red', width=2)
        
        cmap = plt.get_cmap('viridis', len(path))
        for i, (u, v) in enumerate(edges):
            plt.text((pos[u][0] + pos[v][0]) / 2, (pos[u][1] + pos[v][1]) / 2, str(i + 1),
                     color=cmap(i), fontsize=8, ha='center')
        
        start_node = path[0]
        plt.scatter([pos[start_node][0]], [pos[start_node][1]], c='green', s=100, zorder=5)
        plt.text(pos[start_node][0], pos[start_node][1], 'Start', fontsize=12, ha='right', color='green')

    plt.title(title)
    plt.show()

def main():
    num_points_list = [100, 200, 300, 400, 500]
    radius = 0.15
    tsp_times = []
    christofides_times = []
    
    for num_points in num_points_list:
        G, pos = generate_city_graph(num_points, radius)
        
        tsp_time = measure_time('tsp', G, pos)
        christofides_time = measure_time('christofides', G, pos)
        
        tsp_times.append(tsp_time)
        christofides_times.append(christofides_time)
        
        print(f"Points: {num_points}, TSP Time: {tsp_time:.4f}s, Christofides Time: {christofides_time:.4f}s")
    
    plt.plot(num_points_list, tsp_times, label='Traveling Salesman Problem (Approx)')
    plt.plot(num_points_list, christofides_times, label='Christofides')
    plt.xlabel('Number of Points')
    plt.ylabel('Time (seconds)')
    plt.title('TSP Algorithm Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
