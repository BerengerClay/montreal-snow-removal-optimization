import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
from scipy.spatial.distance import cdist
import numpy as np

# Chemin vers votre fichier GeoJSON
geojson_fp = 'Data/geobase.json'

# Charger le fichier GeoJSON
gdf = gpd.read_file(geojson_fp)

# Vérifiez et définissez le système de coordonnées (si nécessaire)
if gdf.crs is None:
    gdf.set_crs(epsg=4326, inplace=True)  # WGS84

# Définir les quartiers d'intérêt
# quartiers_interet = ['Outremont', 'Verdun', 'Anjou', 
#                      'Rivière-des-Prairies-Pointe-aux-Trembles', 
#                      'Le Plateau-Mont-Royal']
quartiers_interet = ['Outremont']

# Filtrer les données par quartier
gdf_filtered = gdf[(gdf['ARR_GCH'].isin(quartiers_interet)) | 
                   (gdf['ARR_DRT'].isin(quartiers_interet))]


def extract_coordinates(gdf):
    """Extract coordinates from the GeoDataFrame."""
    coords = []
    for geom in gdf.geometry:
        if geom.geom_type == 'LineString':
            coords.extend(geom.coords)
        elif geom.geom_type == 'MultiLineString':
            for line in geom:
                coords.extend(line.coords)
    return np.array(coords)

def create_graph(coords):
    """Create a graph with nodes as coordinates and edges as distances."""
    G = nx.Graph()
    for i, coord in enumerate(coords):
        G.add_node(i, pos=coord)
    distances = cdist(coords, coords, 'euclidean')
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            G.add_edge(i, j, weight=distances[i, j])
    return G

def solve_tsp(G):
    """Solve the TSP problem on the graph."""
    tsp_path = nx.approximation.christofides(G, weight='weight')
    return tsp_path

def plot_tsp_path(ax, coords, tsp_path, quartier):
    """Plot the TSP path on the given axes."""
    for i in range(len(tsp_path) - 1):
        start, end = tsp_path[i], tsp_path[i + 1]
        ax.plot([coords[start][0], coords[end][0]], [coords[start][1], coords[end][1]], 'r-')
    ax.set_title(f'TSP Path for {quartier}')

# Créer une figure avec des sous-graphes pour chaque quartier
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for ax, quartier in zip(axes, quartiers_interet):
    subset = gdf_filtered[(gdf_filtered['ARR_GCH'] == quartier) | 
                          (gdf_filtered['ARR_DRT'] == quartier)]
    
    if not subset.empty:
        coords = extract_coordinates(subset)
        G = create_graph(coords)
        tsp_path = solve_tsp(G)
        
        minx, miny, maxx, maxy = subset.total_bounds
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
        subset.plot(ax=ax)
        plot_tsp_path(ax, coords, tsp_path, quartier)
        ax.set_aspect('equal')
    else:
        print(f"Pas de données pour le quartier {quartier}")
        ax.text(0.5, 0.5, f'Pas de données pour {quartier}', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        ax.set_title(f'Segments de {quartier}')
        ax.set_axis_off()

# Désactiver les axes restants s'il y en a plus que le nombre de quartiers
for i in range(len(quartiers_interet), len(axes)):
    axes[i].set_axis_off()

plt.tight_layout()
plt.show()
