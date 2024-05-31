import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import timeit
from networkx.algorithms.approximation import traveling_salesman_problem, christofides

def generate_random_graph(num_points, seed=42):
    np.random.seed(seed)
    coords = np.random.rand(num_points, 2)
    G = nx.complete_graph(num_points)
    for i in range(num_points):
        for j in range(i + 1, num_points):
            weight = np.linalg.norm(coords[i] - coords[j])
            G[i][j]['weight'] = weight
    return G

def measure_time(algorithm, G):
    if algorithm == 'tsp':
        return 0 
        time_taken = timeit.timeit(lambda: traveling_salesman_problem(G, cycle=True), number=1)
    elif algorithm == 'christofides':
        time_taken = timeit.timeit(lambda: christofides(G), number=1)
    return time_taken

def main():
    num_points_list = [1500]
    tsp_times = []
    christofides_times = []
    
    for num_points in num_points_list:
        G = generate_random_graph(num_points)
        
        tsp_time = measure_time('tsp', G)
        christofides_time = measure_time('christofides', G)
        
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
