import ee
import folium
import pandas as pd
import re
import os

# Initialize Earth Engine
try:
    ee.Initialize(project='alphaearth4nefd')
except Exception as e:
    print("GEE not initialized. Please run `ee.Authenticate()` and `ee.Initialize()` first.")
    print(f"Error details: {e}")
    exit(1)

# --- Parameters ---
VALIDATION_ASSET_ID = 'projects/alphaearth4nefd/assets/East_Coast_Small_scale_forests_2025'
TA_BOUNDARIES_ID = 'projects/ee-feiteng0802/assets/SHP_simple_5km_NZTA'
EMBEDDING_DATASET_ID = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
TARGET_TA_NAME = 'Gisborne District'
LCDB_POLYGONS_ID = 'projects/alphaearth4nefd/assets/lcdb_potentialForest'

# Map Center (Gisborne)
MAP_CENTER = [-38.6625, 177.98] 
MAP_ZOOM = 10

def get_cluster_ids_from_recipe(recipe_str):
    if pd.isna(recipe_str) or recipe_str == "None":
        return []
    matches = re.findall(r'cluster_(\d+)', recipe_str)
    return [int(m) for m in matches]

def clean_image(image):
    band_names = image.bandNames()
    new_names = band_names.map(lambda n: ee.String('b').cat(n))
    return image.rename(new_names).toFloat().unmask(0)

def add_ee_layer(self, ee_image_object, vis_params, name, show=True):
    map_id_dict = ee.Image(ee_image_object).getMapId(vis_params)
    folium.raster_layers.TileLayer(
        tiles=map_id_dict['tile_fetcher'].url_format,
        attr='Map Data &copy; <a href="https://earthengine.google.com/">Google Earth Engine</a>',
        name=name,
        overlay=True,
        control=True,
        show=show
    ).add_to(self)

def main():
    print("Starting Map Visualization (Focus on R_L)...")
    
    # 1. Clusters for R_L (Radiata Harvesting - Mature/Large)
    # Using the decoded clusters confirmed by user: 12, 14, 65
    unique_clusters_ga = [12, 14, 65]
    print(f"Clusters for R_L: {unique_clusters_ga}")
    
    # 2. Setup GEE Objects
    ta_boundaries = ee.FeatureCollection(TA_BOUNDARIES_ID)
    validation_region = ta_boundaries.filter(ee.Filter.eq('TA2023_V_1', TARGET_TA_NAME)).geometry()
    
    lcdb_polygons = ee.FeatureCollection(LCDB_POLYGONS_ID)
    lc_filter = ee.Filter.inList('Class_2018', [71, 64, 40, 41, 50, 52])
    analysis_mask = ee.Image().byte().paint(lcdb_polygons.filter(lc_filter), 1).unmask(0).selfMask()
    
    embedding_col = ee.ImageCollection(EMBEDDING_DATASET_ID)
    
    # 3. Train Clusterer
    print("Training Clusterer (10,000 samples, Seed 42)...")
    ALL_TRAINING_YEARS = [2019, 2020, 2021, 2022, 2023]
    NUMBER_OF_CLUSTERS = 100
    TOTAL_SAMPLES = 10000 
    
    multi_year_samples = ee.FeatureCollection([])
    samples_per_year = int(TOTAL_SAMPLES / len(ALL_TRAINING_YEARS))
    
    for year in ALL_TRAINING_YEARS:
        img1 = embedding_col.filterDate(f'{year}-01-01', f'{year+1}-01-01').median()
        img2 = embedding_col.filterDate(f'{year+1}-01-01', f'{year+2}-01-01').median()
        change = clean_image(img2.subtract(img1)).updateMask(analysis_mask)
        samples = change.sample(
            region=ta_boundaries.geometry(),
            scale=300,
            numPixels=samples_per_year,
            seed=42,
            geometries=False,
            tileScale=8
        )
        multi_year_samples = multi_year_samples.merge(samples)
        
    dummy_img = clean_image(embedding_col.filterDate('2022-01-01', '2023-01-01').median())
    band_names = dummy_img.bandNames()
    
    clusterer = ee.Clusterer.wekaKMeans(
        nClusters=NUMBER_OF_CLUSTERS,
        maxIterations=100
    ).train(multi_year_samples, band_names)
    
    print("Clusterer trained.")

    # 4. Generate Maps
    # Locations: Overview (Gisborne), Detail (Mangatu Forest)
    locations = [
        {'name': 'overview', 'center': [-38.6625, 177.98], 'zoom': 10},
        {'name': 'detail', 'center': [-38.30, 177.85], 'zoom': 13}
    ]
    years_to_map = [2022, 2023]
    
    folium.Map.add_ee_layer = add_ee_layer
    
    for loc in locations:
        for year in years_to_map:
            print(f"Generating {loc['name']} Map for {year}...")
            
            # Start map with a neutral tile that we can layer over
            m = folium.Map(location=loc['center'], zoom_start=loc['zoom'], tiles=None)
            
            # Add Basemaps
            folium.TileLayer(
                tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
                attr='Google',
                name='Google Satellite',
                overlay=False,
                control=True
            ).add_to(m)
            
            folium.TileLayer(
                tiles='OpenStreetMap',
                name='OpenStreetMap',
                overlay=False,
                control=True
            ).add_to(m)
            
            # A. Ground Truth (Blue) - Now for BOTH 2022 and 2023
            def add_age(f):
                h_yr = ee.Number(f.get('HarvestYr'))
                age_h99 = ee.Number(f.get('Age_H99'))
                age = ee.Algorithms.If(
                    h_yr.gt(0).And(age_h99.gt(0)),
                    age_h99.add(h_yr.subtract(2020)),
                    0
                )
                return f.set('Age_at_Harvest', age)

            validation_data = ee.FeatureCollection(VALIDATION_ASSET_ID).filterBounds(validation_region).map(add_age)
            
            gt_feats = validation_data.filter(ee.Filter.And(
                ee.Filter.eq('HarvestYr', year),
                ee.Filter.eq('Species_De', 'Radiata'),
                ee.Filter.gt('Age_at_Harvest', 25)
            ))
            
            gt_image = ee.Image().byte().paint(gt_feats, 1).visualize(palette=['0000FF'], opacity=0.8)
            m.add_ee_layer(gt_image, {}, f'Ground Truth {year} R_L (Blue)', True)
            
            # B. Prediction (GA Clusters)
            start_year = year - 1
            img1 = embedding_col.filterDate(f'{start_year}-01-01', f'{start_year+1}-01-01').median()
            img2 = embedding_col.filterDate(f'{start_year+1}-01-01', f'{start_year+2}-01-01').median()
            change_vector = clean_image(img2.subtract(img1)).clip(validation_region)
            
            clusters = change_vector.updateMask(analysis_mask).cluster(clusterer)
            
            # Combine [12, 14, 65]
            combined_mask = ee.Image(0)
            for cid in unique_clusters_ga:
                combined_mask = combined_mask.Or(clusters.eq(cid))
            
            color = 'FF0000' if year == 2022 else '00FFFF' # Red for 2022, Cyan for 2023
            name_color = 'Red' if year == 2022 else 'Cyan'
            
            combined_vis = combined_mask.selfMask().visualize(palette=[color], opacity=0.7)
            m.add_ee_layer(combined_vis, {}, f'Predicted {year} R_L ({name_color})', True)
            
            # Add Layer Control (collapsed=False to make toggle visible)
            folium.LayerControl(collapsed=False).add_to(m)
            
            os.makedirs("code/plots", exist_ok=True)
            out_file = f"code/plots/gisborne_{loc['name']}_{year}.html"
            m.save(out_file)
            print(f"Saved map to {out_file}")

if __name__ == '__main__':
    main()
