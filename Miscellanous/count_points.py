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

def count_coordinates(gdf):
    coords = []
    for geom in gdf.geometry:
        if geom.geom_type == 'LineString':
            coords.extend(geom.coords)
        elif geom.geom_type == 'MultiLineString':
            for line in geom:
                coords.extend(line.coords)
    return len(coords)

points_count = {}
for quartier in quartiers_interet:
    subset = gdf_filtered[(gdf_filtered['ARR_GCH'] == quartier) | 
                          (gdf_filtered['ARR_DRT'] == quartier)]
    points_count[quartier] = count_coordinates(subset)

for quartier, count in points_count.items():
    print(f"{quartier}: {count} points")
