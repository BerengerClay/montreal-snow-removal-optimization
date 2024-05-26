import geopandas as gpd
import matplotlib.pyplot as plt

geojson_fp = 'Data/geobase.json'

gdf = gpd.read_file(geojson_fp)

# print(gdf.head())

if gdf.crs is None:
    gdf.set_crs(epsg=4326, inplace=True)

quartiers_interet = ['Outremont', 'Verdun', 'Anjou', 
                     'Rivière-des-Prairies-Pointe-aux-Trembles', 
                     'Le Plateau-Mont-Royal']

gdf_filtered = gdf[(gdf['ARR_GCH'].isin(quartiers_interet)) | 
                   (gdf['ARR_DRT'].isin(quartiers_interet))]

# print(gdf_filtered['ARR_GCH'].unique())
# print(gdf_filtered['ARR_DRT'].unique())

nrows = 2
ncols = 3

fig, axes = plt.subplots(nrows, ncols, figsize=(15, 10))

axes = axes.flatten()

for ax, quartier in zip(axes, quartiers_interet):
    subset = gdf_filtered[(gdf_filtered['ARR_GCH'] == quartier) | 
                          (gdf_filtered['ARR_DRT'] == quartier)]
    
    if not subset.empty:
        minx, miny, maxx, maxy = subset.total_bounds
        if all(map(lambda x: x is not None and x != float('inf') and x != float('-inf'), [minx, miny, maxx, maxy])):
            ax.set_xlim(minx, maxx)
            ax.set_ylim(miny, maxy)
            subset.plot(ax=ax)
            ax.set_title(f'Segments de {quartier}')
            ax.set_aspect('equal')
        else:
            print(f"Les limites pour le quartier {quartier} ne sont pas valides : {minx}, {miny}, {maxx}, {maxy}")
    else:
        print(f"Pas de données pour le quartier {quartier}")
        ax.text(0.5, 0.5, f'Pas de données pour {quartier}', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        ax.set_title(f'Segments de {quartier}')
        ax.set_axis_off()

for i in range(len(quartiers_interet), len(axes)):
    axes[i].set_axis_off()

plt.tight_layout()
plt.show()
