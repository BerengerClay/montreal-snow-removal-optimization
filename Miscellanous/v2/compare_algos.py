import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import timeit
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit
from networkx.algorithms.approximation import traveling_salesman_problem, christofides
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

def make_complete_graph(G):
    complete_graph = nx.complete_graph(len(G.nodes()))
    mapping = {i: node for i, node in enumerate(G.nodes())}
    complete_graph = nx.relabel_nodes(complete_graph, mapping)
    
    pos = nx.get_node_attributes(G, 'pos')
    for u, v in complete_graph.edges():
        if not G.has_edge(u, v):
            dist = np.linalg.norm(np.array(u) - np.array(v))
            complete_graph.add_edge(u, v, weight=dist)
        else:
            complete_graph[u][v]['weight'] = G[u][v]['weight']
    
    return complete_graph

def measure_time(G):
    complete_G = make_complete_graph(G)
    tsp_time = timeit.timeit(lambda: traveling_salesman_problem(complete_G, weight='weight', cycle=True), number=1)
    christofides_time = timeit.timeit(lambda: christofides(complete_G, weight='weight'), number=1)
    return tsp_time, christofides_time

def exponential(x, a, b, c):
    return a * np.exp(b * x) + c

def polynomial_regression(edge_counts, times, degree):
    poly_features = PolynomialFeatures(degree=degree)
    X_poly = poly_features.fit_transform(edge_counts)
    poly_model = LinearRegression().fit(X_poly, times)
    return poly_model, poly_features

def main():
    grid_sizes = [int(i**1.4) for i in range(2,10)]
    edge_probs = [0.6]

    results = []
    edge_node_ratios = []

    for grid_size in tqdm(grid_sizes):
        for edge_prob in edge_probs:
            extra_edges = int(1/100 * grid_size ** 2)
            G = generate_city_graph(grid_size, edge_prob, extra_edges)
            tsp_time, christofides_time = measure_time(G)
            edge_count = len(G.edges())
            node_count = len(G.nodes())
            edge_node_ratio = edge_count / node_count
            results.append((grid_size, edge_count, tsp_time, christofides_time))
            edge_node_ratios.append(edge_node_ratio)

    edge_counts = np.array([r[1] for r in results]).reshape(-1, 1)
    tsp_times = np.array([r[2] for r in results])
    christofides_times = np.array([r[3] for r in results])

    tsp_poly_model, tsp_poly_features = polynomial_regression(edge_counts, tsp_times, degree=2)
    christofides_poly_model, christofides_poly_features = polynomial_regression(edge_counts, christofides_times, degree=2)

    tsp_exp_params, _ = curve_fit(exponential, edge_counts.ravel(), tsp_times, maxfev=10000)
    christofides_exp_params, _ = curve_fit(exponential, edge_counts.ravel(), christofides_times, maxfev=10000)

    future_edge_counts = np.arange(edge_counts.max(), 15001).reshape(-1, 1)
    
    tsp_poly_future_times = tsp_poly_model.predict(tsp_poly_features.transform(future_edge_counts))
    christofides_poly_future_times = christofides_poly_model.predict(christofides_poly_features.transform(future_edge_counts))
    
    tsp_exp_future_times = exponential(future_edge_counts.ravel(), *tsp_exp_params)
    christofides_exp_future_times = exponential(future_edge_counts.ravel(), *christofides_exp_params)

    tsp_poly_mse = mean_squared_error(tsp_times, tsp_poly_model.predict(tsp_poly_features.transform(edge_counts)))
    tsp_exp_mse = mean_squared_error(tsp_times, exponential(edge_counts.ravel(), *tsp_exp_params))
    
    christofides_poly_mse = mean_squared_error(christofides_times, christofides_poly_model.predict(christofides_poly_features.transform(edge_counts)))
    christofides_exp_mse = mean_squared_error(christofides_times, exponential(edge_counts.ravel(), *christofides_exp_params))
    
    tsp_best_future_times = tsp_poly_future_times if tsp_poly_mse < tsp_exp_mse else tsp_exp_future_times
    christofides_best_future_times = christofides_poly_future_times if christofides_poly_mse < christofides_exp_mse else christofides_exp_future_times

    plt.plot(edge_counts, tsp_times, color='red', label='TSP')
    plt.plot(edge_counts, christofides_times, color='blue', label='Christofides')
    plt.plot(future_edge_counts, tsp_best_future_times, color='red', linestyle='--', label='TSP Prediction')
    plt.plot(future_edge_counts, christofides_best_future_times, color='blue', linestyle='--', label='Christofides Prediction')

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
    edge_node_ratios = main()
