import ee
import math

# Initialize Earth Engine
try:
    ee.Initialize(project='alphaearth4nefd')
except Exception as e:
    print("GEE not initialized. Please run `ee.Authenticate()` and `ee.Initialize()` first.")
    # In a real script we might raise, but here we let the user know.

# --- User Parameters ---
START_YEAR = 2019
END_YEAR = 2024
TA_BOUNDARIES_ID = 'projects/ee-feiteng0802/assets/SHP_simple_5km_NZTA'
TA_NAME_PROPERTY = 'TA2023_V_1'
NUMBER_OF_CLUSTERS = 200  # Changed to 200 as requested
SIMILARITY_THRESHOLD = 0.95
EMBEDDING_DATASET_ID = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
LCDB_POLYGONS_ID = 'projects/alphaearth4nefd/assets/lcdb_potentialForest'
EXPORT_FOLDER = 'GEE_Exports_Universal_200'

# [Critical] Define all years for training to ensure the clusterer sees all variations
ALL_TRAINING_YEARS = [2019, 2020, 2021, 2022, 2023]

def clean_image(image):
    """Cleans Embedding image bands and handles masking."""
    # Rename bands to b0, b1, ... to avoid issues
    band_names = image.bandNames()
    new_names = band_names.map(lambda n: ee.String('b').cat(n))
    return image.rename(new_names).toFloat().unmask(0)

def main():
    print(f"Starting GEE Clustering Export (K={NUMBER_OF_CLUSTERS})...")
    
    embedding_col = ee.ImageCollection(EMBEDDING_DATASET_ID)
    ta_boundaries = ee.FeatureCollection(TA_BOUNDARIES_ID)
    
    # --- Prepare LCDB Mask ---
    lcdb_polygons = ee.FeatureCollection(LCDB_POLYGONS_ID)
    class_column = 'Class_2018'
    # Forest classes: 71 (Exotic Forest), 64 (Forest - Harvested), 
    # 40 (Indigenous Forest), 41 (Broadleaved Indigenous Hardwoods),
    # 50 (Fernland), 52 (Manuka and/or Kanuka)
    lc_filter = ee.Filter.Or(
        ee.Filter.eq(class_column, 71), ee.Filter.eq(class_column, 64),
        ee.Filter.eq(class_column, 40), ee.Filter.eq(class_column, 41),
        ee.Filter.eq(class_column, 50), ee.Filter.eq(class_column, 52)
    )
    analysis_polygons = lcdb_polygons.filter(lc_filter)
    # Create a binary mask from polygons
    analysis_mask = ee.Image().byte().paint(analysis_polygons, 1).unmask(0).selfMask()

    # --- Step 1: Prepare Multi-Year Training Data ---
    print('Step 1: Preparing Multi-Year Training Data...')
    
    # We need to map over a client-side list to create a server-side FeatureCollection
    # But ee.FeatureCollection(list) expects features. 
    # Better to iterate and merge.
    
    multi_year_samples = ee.FeatureCollection([])
    
    # Total samples we want
    TOTAL_SAMPLES = 10000
    samples_per_year = int(TOTAL_SAMPLES / len(ALL_TRAINING_YEARS))
    
    for year in ALL_TRAINING_YEARS:
        img1 = embedding_col.filterDate(f'{year}-01-01', f'{year+1}-01-01').median()
        img2 = embedding_col.filterDate(f'{year+1}-01-01', f'{year+2}-01-01').median()
        
        # Calculate change vector
        change_vector = clean_image(img2.subtract(img1))
        
        # Apply mask
        masked_change = change_vector.updateMask(analysis_mask)
        
        # Sample
        samples = masked_change.sample(
            region=ta_boundaries.geometry(),
            scale=300, # Coarser scale for training is usually fine and faster
            numPixels=samples_per_year,
            seed=42,
            geometries=False,
            tileScale=8
        )
        multi_year_samples = multi_year_samples.merge(samples)
        
    print('Training samples prepared (lazy evaluation).')

    # --- Step 2: Train Universal Clusterer ---
    print('Step 2: Training Universal K-Means Clusterer...')
    
    # Get band names from a dummy image
    dummy_img = clean_image(
        embedding_col.filterDate('2022-01-01', '2023-01-01').median()
    )
    band_names = dummy_img.bandNames()
    
    # Train Weka K-Means
    clusterer = ee.Clusterer.wekaKMeans(
        nClusters=NUMBER_OF_CLUSTERS,
        maxIterations=100
    ).train(multi_year_samples, band_names)
    
    print('Clusterer defined.')

    # --- Step 3: Year-by-Year Analysis & Export ---
    print('Step 3: Creating Export Tasks...')
    
    for year in range(START_YEAR, END_YEAR):
        year_next = year + 1
        year_period_string = f"{year}-{year_next}"
        
        # 1. Prepare images
        img1 = embedding_col.filterDate(f'{year}-01-01', f'{year_next}-01-01').median()
        img2 = embedding_col.filterDate(f'{year_next}-01-01', f'{year_next+1}-01-01').median()
        
        change_vector = clean_image(img2.subtract(img1))
        
        # 2. Apply LCDB Mask
        masked_change_vector = change_vector.updateMask(analysis_mask)
        
        # 3. Similarity Mask (filter small changes)
        # Use raw images for dot product to capture magnitude correctly
        dot_prod = img1.multiply(img2).reduce(ee.Reducer.sum())
        similarity_mask = dot_prod.lt(SIMILARITY_THRESHOLD)
        
        # 4. Apply Clusterer
        change_clusters = masked_change_vector.cluster(clusterer)
        
        # 5. Apply Similarity Mask
        final_clusters = change_clusters.updateMask(similarity_mask)
        
        # 6. Calculate Area (Hectares)
        # Pixel area in m^2, divide by 10000 for hectares
        area_image = ee.Image.pixelArea().divide(10000).addBands(final_clusters)
        
        # 7. Reduce Regions (Zonal Statistics)
        # Group by cluster ID (index 1) and sum area (index 0)
        results = area_image.reduceRegions(
            collection=ta_boundaries,
            reducer=ee.Reducer.sum().group(groupField=1, groupName='cluster'),
            scale=100, # 100m resolution for area calculation
            tileScale=8
        )
        
        # 8. Format for Export (Pivot the groups to columns)
        # This part is tricky in Python client because we can't write complex mapped functions 
        # as easily as JS if they depend on server-side objects in a specific way, 
        # but the logic holds.
        
        def format_feature(feature):
            properties = {
                'ta_name': feature.get(TA_NAME_PROPERTY),
                'year_period': year_period_string,
                'unit': 'hectares'
            }
            
            # Get the groups list
            groups = ee.List(feature.get('groups'))
            # Handle null groups (no clusters found in TA)
            groups = ee.List(ee.Algorithms.If(groups, groups, []))
            
            # Create a dictionary of 0.0 for all clusters
            # In Python client we construct the list of keys
            cluster_indices = ee.List.sequence(0, NUMBER_OF_CLUSTERS - 1)
            cluster_keys = cluster_indices.map(lambda i: ee.String('cluster_').cat(ee.Number(i).format('%d')))
            base_values = ee.List.repeat(0.0, NUMBER_OF_CLUSTERS)
            base_dict = ee.Dictionary.fromLists(cluster_keys, base_values)
            
            # Function to update the dictionary with actual values
            def update_dict(group_item, acc_dict):
                group_dict = ee.Dictionary(group_item)
                cluster_id = ee.String('cluster_').cat(ee.Number(group_dict.get('cluster')).format('%d'))
                area = group_dict.get('sum')
                return ee.Dictionary(acc_dict).set(cluster_id, area)
            
            updated_dict = groups.iterate(update_dict, base_dict)
            
            return ee.Feature(None, ee.Dictionary(properties).combine(updated_dict))

        final_table = results.map(format_feature)
        
        # 9. Create Export Task
        selectors = ['ta_name', 'year_period', 'unit'] + [f'cluster_{i}' for i in range(NUMBER_OF_CLUSTERS)]
        
        output_name = f"NZ_TA_Universal_{NUMBER_OF_CLUSTERS}Clusters_{year_period_string}"
        
        task = ee.batch.Export.table.toDrive(
            collection=final_table,
            description=output_name,
            folder=EXPORT_FOLDER,
            fileNamePrefix=output_name,
            fileFormat='CSV',
            selectors=selectors
        )
        
        task.start()
        print(f"Task started: {output_name} (ID: {task.id})")

if __name__ == '__main__':
    main()
