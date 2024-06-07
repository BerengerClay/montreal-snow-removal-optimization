import geopandas as gpd
from geopy.distance import geodesic

def load_and_prepare_data(geojson_fp, street_name, quartier):
    gdf = gpd.read_file(geojson_fp)

    if gdf.crs is None:
        gdf.set_crs(epsg=4326, inplace=True)

    gdf_filtered = gdf[(gdf['NOM_VOIE'].str.contains(street_name, case=False, na=False)) & 
                       ((gdf['ARR_GCH'].str.lower() == quartier.lower()) | 
                        (gdf['ARR_DRT'].str.lower() == quartier.lower()))]
    
    return gdf_filtered

def calculate_street_length(gdf):
    total_length = 0

    for _, row in gdf.iterrows():
        coords = list(row.geometry.coords)
        street_name = row['NOM_VOIE']
        for i in range(len(coords) - 1):
            u = (coords[i][1], coords[i][0])
            v = (coords[i + 1][1], coords[i + 1][0])
            segment_length = geodesic(u, v).meters
            total_length += segment_length
            print(f"Segment of {street_name} from {u} to {v}: {segment_length:.2f} meters")
    
    return total_length

def main():
    geojson_fp = 'Data/geobase.json'
    street_name = 'Willowdale'  # Nom de la rue à vérifier
    quartier = 'Outremont'  # Nom du quartier à vérifier

    gdf_filtered = load_and_prepare_data(geojson_fp, street_name, quartier)
    
    if not gdf_filtered.empty:
        total_length = calculate_street_length(gdf_filtered)
        print(f"The total length of {street_name} in {quartier} is {total_length:.2f} meters")
    else:
        print(f"No segments found for street {street_name} in {quartier}")

if __name__ == "__main__":
    main()
