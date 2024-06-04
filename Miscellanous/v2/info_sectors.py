import geopandas as gpd

geojson_fp = 'Data/geobase.json'

gdf = gpd.read_file(geojson_fp)

if gdf.crs is None:
    gdf.set_crs(epsg=4326, inplace=True)

quartiers_interet = ['Outremont', 'Verdun', 'Anjou', 
                     'Rivière-des-Prairies-Pointe-aux-Trembles', 
                     'Le Plateau-Mont-Royal']

gdf_filtered = gdf[(gdf['ARR_GCH'].isin(quartiers_interet)) | 
                   (gdf['ARR_DRT'].isin(quartiers_interet))]

def count_edges_and_nodes(gdf):
    edge_count = 0
    node_set = set()
    for geom in gdf.geometry:
        if geom.geom_type == 'LineString':
            coords = list(geom.coords)
            edge_count += len(coords) - 1
            node_set.update(coords)
        elif geom.geom_type == 'MultiLineString':
            for line in geom:
                coords = list(line.coords)
                edge_count += len(coords) - 1
                node_set.update(coords)
    return edge_count, len(node_set)

stats = {}
for quartier in quartiers_interet:
    subset = gdf_filtered[(gdf_filtered['ARR_GCH'] == quartier) | 
                          (gdf_filtered['ARR_DRT'] == quartier)]
    edges, nodes = count_edges_and_nodes(subset)
    ratio = edges / nodes if nodes != 0 else 0
    stats[quartier] = {'edges': edges, 'nodes': nodes, 'ratio': ratio}

with open('Miscellanous/info_sectors.txt', 'w') as file:
    for quartier, data in stats.items():
        file.write(f"{quartier}:\n")
        file.write(f"  Nombre de nœuds: {data['nodes']}\n")
        file.write(f"  Nombre d'arêtes: {data['edges']}\n")
        file.write(f"  Ratio arêtes/nœuds: {data['ratio']:.2f}\n")

print("Les résultats ont été enregistrés dans 'nb_edges_nodes.txt'")
