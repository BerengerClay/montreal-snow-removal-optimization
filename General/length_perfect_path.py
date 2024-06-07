import geopandas as gpd
from geopy.distance import geodesic

def load_and_prepare_data(geojson_fp, quartiers_interet):
    gdf = gpd.read_file(geojson_fp)

    if gdf.crs is None:
        gdf.set_crs(epsg=4326, inplace=True)

    gdf_filtered = gdf[gdf['ARR_GCH'].isin(quartiers_interet) | gdf['ARR_DRT'].isin(quartiers_interet)]
    return gdf_filtered

def calculate_total_road_length(gdf):
    total_length = 0

    for _, row in gdf.iterrows():
        coords = list(row.geometry.coords)
        for i in range(len(coords) - 1):
            point1 = (coords[i][1], coords[i][0])
            point2 = (coords[i + 1][1], coords[i + 1][0])
            segment_length = geodesic(point1, point2).meters
            total_length += segment_length

    return total_length

def main():
    geojson_fp = 'Data/geobase.json'
    quartiers_interet = ['Outremont', 'Verdun', 'Anjou', 'Rivière-des-Prairies-Pointe-aux-Trembles', 'Le Plateau-Mont-Royal']

    gdf_filtered = load_and_prepare_data(geojson_fp, quartiers_interet)
    
    for quartier in quartiers_interet:
        gdf_quartier = gdf_filtered[(gdf_filtered['ARR_GCH'] == quartier) | (gdf_filtered['ARR_DRT'] == quartier)]
        total_length = calculate_total_road_length(gdf_quartier)
        print(f'The total length of roads in {quartier} is {total_length:.2f} meters')

if __name__ == "__main__":
    main()
