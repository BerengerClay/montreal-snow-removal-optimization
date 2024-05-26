import geopandas as gpd

# Chemin vers votre fichier GeoJSON
geojson_fp = 'Data/geobase.json'

# Charger le fichier GeoJSON
gdf = gpd.read_file(geojson_fp)

# Vérifiez et définissez le système de coordonnées (si nécessaire)
if gdf.crs is None:
    gdf.set_crs(epsg=4326, inplace=True)  # WGS84

# Définir les quartiers d'intérêt
quartiers_interet = ['Outremont', 'Verdun', 'Anjou', 
                     'Rivière-des-Prairies-Pointe-aux-Trembles', 
                     'Le Plateau-Mont-Royal']

# Filtrer les données par quartier
gdf_filtered = gdf[(gdf['ARR_GCH'].isin(quartiers_interet)) | 
                   (gdf['ARR_DRT'].isin(quartiers_interet))]

# Fonction pour extraire et compter les coordonnées
def count_coordinates(gdf):
    coords = []
    for geom in gdf.geometry:
        if geom.geom_type == 'LineString':
            coords.extend(geom.coords)
        elif geom.geom_type == 'MultiLineString':
            for line in geom:
                coords.extend(line.coords)
    return len(coords)

# Compter les points dans chaque quartier
points_count = {}
for quartier in quartiers_interet:
    subset = gdf_filtered[(gdf_filtered['ARR_GCH'] == quartier) | 
                          (gdf_filtered['ARR_DRT'] == quartier)]
    points_count[quartier] = count_coordinates(subset)

# Afficher le nombre de points pour chaque quartier
for quartier, count in points_count.items():
    print(f"{quartier}: {count} points")
