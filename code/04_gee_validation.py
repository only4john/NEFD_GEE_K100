import ee
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
YEARS_TO_CALCULATE = [2020, 2021, 2022, 2023]

# LCDB Mask (Same as training)
LCDB_POLYGONS_ID = 'projects/alphaearth4nefd/assets/lcdb_potentialForest'

# --- Helper Functions ---
def get_cluster_ids_from_recipe(recipe_str):
    """Parses 'cluster_12(100%), cluster_45(80%)' into [12, 45]."""
    if pd.isna(recipe_str) or recipe_str == "None":
        return []
    # Find all 'cluster_XX' patterns
    matches = re.findall(r'cluster_(\d+)', recipe_str)
    return [int(m) for m in matches]

def clean_image(image):
    band_names = image.bandNames()
    new_names = band_names.map(lambda n: ee.String('b').cat(n))
    return image.rename(new_names).toFloat().unmask(0)

def calculate_age_at_year(feature, year):
    """Calculate forest age at a specific year."""
    age_2025 = ee.Number(feature.get('Age_2025'))
    years_back = 2025 - year
    age_at_year = age_2025.add(years_back)
    return ee.Algorithms.If(age_2025.gt(0), age_at_year, 0)

def validate_target(target_name, target_clusters, validation_data, validation_region, 
                   clusterer, embedding_col, analysis_mask, filter_function, years, lag_offset=0):
    """
    Generic validation function for any target.
    
    Args:
        target_name: Name of the target (e.g., 'R_L', 'R_S')
        target_clusters: List of cluster IDs for this target
        validation_data: GEE FeatureCollection with ground truth
        validation_region: GEE Geometry for the validation area
        clusterer: Trained GEE clusterer
        embedding_col: GEE ImageCollection for embeddings
        analysis_mask: GEE Image for forest mask
        filter_function: Function to filter validation data for each year
        years: List of years to validate
        lag_offset: if -1, compares Pred(T-1) vs GT(T)
    
    Returns:
        List of dictionaries with validation results
    """
    print(f"\n=== {target_name} Validation {'(Lagged)' if lag_offset != 0 else ''} ===")
    print(f"{'Year':<5} | {'Recall':<6} | {'GT (ha)':<7} | {'Found (ha)':<10}")
    print("-" * 40)
    
    results = []
    for year in years:
        # A. Ground Truth Raster (Using paint which is usually more efficient than reduceToImage)
        gt_feats = filter_function(validation_data, year)
        gt_raster = ee.Image().byte().paint(gt_feats, 1).gt(0).unmask(0).rename('gt').clip(validation_region)
                            
        # B. Prediction Raster (possibly from a different year)
        pred_year = year + lag_offset
        img1 = embedding_col.filterDate(f'{pred_year-1}-01-01', f'{pred_year}-01-01').median()
        img2 = embedding_col.filterDate(f'{pred_year}-01-01', f'{pred_year+1}-01-01').median()
        
        # CRITICAL: Same exact processing as training
        change_vector = clean_image(img2.subtract(img1)).clip(validation_region)
        
        clusters = change_vector.updateMask(analysis_mask).cluster(clusterer)
        
        # Remap target clusters to binary mask
        pred_mask = clusters.remap(
            target_clusters, 
            ee.List.repeat(1, len(target_clusters)), 
            0
        ).rename('pred')
        
        # C. Calculate Stats
        combined = gt_raster.addBands(pred_mask).addBands(analysis_mask.rename('mask'))
        combined = combined.updateMask(combined.select('mask'))
        
        matched = pred_mask.And(gt_raster).rename('matched')
        combined = combined.addBands(matched)
        
        stats = combined.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=validation_region,
            scale=30,
            maxPixels=1e10,
            tileScale=16
        )
        
        # Retry logic for robust connection
        for attempt in range(3):
            try:
                val = stats.getInfo()
                gt_px = val.get('gt', 0)
                match_px = val.get('matched', 0)
                
                gt_ha = gt_px * 0.09
                match_ha = match_px * 0.09
                
                recall = match_ha / gt_ha if gt_ha > 0 else 0
                
                print(f"{year:<5} | {recall:<6.4f} | {gt_ha:<7.1f} | {match_ha:<10.1f}")
                results.append({
                    'Year': year, 
                    'Recall': recall, 
                    'GT_ha': gt_ha, 
                    'Found_ha': match_ha
                })
                break # Success
            except Exception as e:
                print(f"{year} | Attempt {attempt+1} failed: {e}")
                if attempt == 2:
                    results.append({
                        'Year': year, 
                        'Recall': 0, 
                        'GT_ha': 0, 
                        'Found_ha': 0
                    })
                import time
                time.sleep(5)
    
    return results

def filter_r_l_harvest(validation_data, year):
    """Filter for Radiata harvest (old forests, age > 25) in given year."""
    def add_age(f):
        h_yr = ee.Number(f.get('HarvestYr'))
        age_h99 = ee.Number(f.get('Age_H99'))
        age = ee.Algorithms.If(
            h_yr.gt(0).And(age_h99.gt(0)),
            age_h99.add(h_yr.subtract(2020)),
            0
        )
        return f.set('Age_at_Harvest', age)
    
    return validation_data.map(add_age).filter(ee.Filter.And(
        ee.Filter.eq('HarvestYr', year),
        ee.Filter.eq('Species_De', 'Radiata'),
        ee.Filter.gt('Age_at_Harvest', 25)
    ))

def filter_r_s_replantation(validation_data, year):
    """Filter for Radiata replantation in given year."""
    return validation_data.filter(ee.Filter.And(
        ee.Filter.eq('Stocked_Ar', year),
        ee.Filter.eq('Species_De', 'Radiata')
    ))

def main():
    print("Starting Validation Analysis...")
    
    # 1. Load Correlation Results to get Clusters
    csv_path = "cluster_activity_correlation_trusted.csv"
    if not os.path.exists(csv_path):
        csv_path = "code/cluster_activity_correlation_trusted.csv"
    if not os.path.exists(csv_path):
        print(f"Error: cluster_activity_correlation_trusted.csv not found.")
        return

    df = pd.read_csv(csv_path)
    df.set_index('Target', inplace=True)
    
    # Extract clusters for R_L and R_S
    target_l = 'delta_Abrupt_R_L'
    target_s = 'delta_Abrupt_R_S'
    
    clusters_l = []
    clusters_s = []
    
    if target_l in df.index:
        recipe = df.loc[target_l, 'Recipe']
        clusters_l = get_cluster_ids_from_recipe(recipe)
        print(f"Clusters for {target_l}: {clusters_l}")
        
    if target_s in df.index:
        recipe = df.loc[target_s, 'Recipe']
        clusters_s = get_cluster_ids_from_recipe(recipe)
        print(f"Clusters for {target_s}: {clusters_s}")

    # 2. Setup GEE Objects
    ta_boundaries = ee.FeatureCollection(TA_BOUNDARIES_ID)
    validation_region = ta_boundaries.filter(ee.Filter.eq('TA2023_V_1', TARGET_TA_NAME)).geometry()
    
    lcdb_polygons = ee.FeatureCollection(LCDB_POLYGONS_ID)
    lc_filter = ee.Filter.inList('Class_2018', [71, 64, 40, 41, 50, 52])
    analysis_mask = ee.Image().byte().paint(lcdb_polygons.filter(lc_filter), 1).unmask(0).selfMask()
    
    embedding_col = ee.ImageCollection(EMBEDDING_DATASET_ID)
    validation_data = ee.FeatureCollection(VALIDATION_ASSET_ID).filterBounds(validation_region)

    # 3. Re-train Clusterer
    print("\nRe-training Clusterer (this may take a moment)...")
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

    # 4. Run Validations
    # 1. R_L (Harvest) - Standard
    years_l = [2020, 2021, 2022, 2023]
    df_l = validate_target("R_L (Harvest)", clusters_l, validation_data, validation_region, 
                          clusterer, embedding_col, analysis_mask, filter_r_l_harvest, years_l)
    pd.DataFrame(df_l).to_csv('code/validation_recall_report_R_L.csv', index=False)
    
    # 2. R_S (Replantation) - Standard (Skip or limited for speed)
    years_s = [2022] # Just one for reference
    df_s = validate_target("R_S (Standard)", clusters_s, validation_data, validation_region, 
                          clusterer, embedding_col, analysis_mask, filter_r_s_replantation, years_s)
    pd.DataFrame(df_s).to_csv('code/validation_recall_report_R_S.csv', index=False)

    # 3. R_S (Replantation) - Lagged (Pred T-1 vs GT T)
    # Predicted change in T-1 (usually harvest) leads to replanting in T
    years_s_lag = [2020, 2021, 2022, 2023]
    df_s_lag = validate_target("R_S (Lagged)", clusters_l, validation_data, validation_region, 
                              clusterer, embedding_col, analysis_mask, filter_r_s_replantation, years_s_lag, lag_offset=-1)
    pd.DataFrame(df_s_lag).to_csv('code/validation_recall_report_R_S_Lagged.csv', index=False)

    print("\nSaved R_L report to code/validation_recall_report_R_L.csv")
    print("Saved R_S report to code/validation_recall_report_R_S.csv")
    print("Saved R_S Lagged report to code/validation_recall_report_R_S_Lagged.csv")

if __name__ == '__main__':
    main()
