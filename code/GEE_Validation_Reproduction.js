/**
 * GEE Validation Reproduction Script (Lagged R_S Support)
 * Paste this into the Google Earth Engine (GEE) Code Editor.
 * 
 * Objective: 
 * 1. Reproduce R_L (Harvest) Recall.
 * 2. Evaluate R_S (Replant) Lagged Recall (Pred Year T vs GT Year T+1).
 */

var VALIDATION_ASSET_ID = 'projects/alphaearth4nefd/assets/East_Coast_Small_scale_forests_2025';
var TA_BOUNDARIES_ID = 'projects/ee-feiteng0802/assets/SHP_simple_5km_NZTA';
var EMBEDDING_DATASET_ID = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL";
var TARGET_TA_NAME = 'Gisborne District';
var LCDB_POLYGONS_ID = 'projects/alphaearth4nefd/assets/lcdb_potentialForest';

// 1. Setup
var ta_boundaries = ee.FeatureCollection(TA_BOUNDARIES_ID);
var validation_region = ta_boundaries.filter(ee.Filter.eq('TA2023_V_1', TARGET_TA_NAME)).geometry();
var embedding_col = ee.ImageCollection(EMBEDDING_DATASET_ID);

var lcdb_polygons = ee.FeatureCollection(LCDB_POLYGONS_ID);
var lc_filter = ee.Filter.inList('Class_2018', [71, 64, 40, 41, 50, 52]);
var analysis_mask = ee.Image().byte().paint(lcdb_polygons.filter(lc_filter), 1).unmask(0).selfMask();

var clean_image = function (image) {
    return image.rename(image.bandNames().map(function (n) { return ee.String('b').cat(n); })).toFloat().unmask(0);
};

// 2. Training (Strictly match 04_gee_validation.py)
print("Training Clusterer...");
var ALL_TRAINING_YEARS = [2019, 2020, 2021, 2022, 2023];
var multi_year_samples = ee.FeatureCollection([]);
ALL_TRAINING_YEARS.forEach(function (year) {
    var img1 = embedding_col.filterDate(year + '-01-01', (year + 1) + '-01-01').median();
    var img2 = embedding_col.filterDate((year + 1) + '-01-01', (year + 2) + '-01-01').median();
    var change = clean_image(img2.subtract(img1)).updateMask(analysis_mask);
    var samples = change.sample({ region: ta_boundaries.geometry(), scale: 300, numPixels: 2000, seed: 42, geometries: false });
    multi_year_samples = multi_year_samples.merge(samples);
});
var clusterer = ee.Clusterer.wekaKMeans(100, 100).train(multi_year_samples);

// 3. Map Setup
Map.setCenter(177.98, -38.6625, 10);
Map.setOptions('HYBRID');

/**
 * Visualizes Prediction vs Ground Truth with optional lag
 * @param {number} pred_year - The year of detected change (T)
 * @param {number} gt_year - The year of ground truth (T or T+1)
 * @param {string} type - 'Harvest' (R_L) or 'Replant' (R_S)
 */
var compare = function (pred_year, gt_year, type, color) {
    var suffix = (pred_year === gt_year) ? '' : ' (Lagged)';
    var name = type + ' ' + pred_year + ' vs GT ' + gt_year + suffix;

    // Prediction T (Change T-1 to T)
    var i1 = embedding_col.filterDate((pred_year - 1) + '-01-01', pred_year + '-01-01').median();
    var i2 = embedding_col.filterDate(pred_year + '-01-01', (pred_year + 1) + '-01-01').median();
    var clusters = clean_image(i2.subtract(i1)).updateMask(analysis_mask).clip(validation_region).cluster(clusterer);

    // Recipe Clusters
    var target_clusters = (type === 'Harvest') ? [12, 14, 65] : [12];
    var pred_mask = clusters.remap(target_clusters, ee.List.repeat(1, target_clusters.length), 0).selfMask();

    // GT
    var validation_data = ee.FeatureCollection(VALIDATION_ASSET_ID).filterBounds(validation_region);
    var gt_feats;
    if (type === 'Harvest') {
        gt_feats = validation_data.map(function (f) {
            var h_yr = ee.Number(f.get('HarvestYr'));
            var age_h99 = ee.Number(f.get('Age_H99'));
            var age = ee.Algorithms.If(h_yr.gt(0).And(age_h99.gt(0)), age_h99.add(h_yr.subtract(2020)), 0);
            return f.set('Age_at_Harvest', age);
        }).filter(ee.Filter.and(ee.Filter.eq('HarvestYr', gt_year), ee.Filter.eq('Species_De', 'Radiata'), ee.Filter.gt('Age_at_Harvest', 25)));
    } else {
        gt_feats = validation_data.filter(ee.Filter.and(ee.Filter.eq('Stocked_Ar', gt_year), ee.Filter.eq('Species_De', 'Radiata')));
    }

    var gt_image = ee.Image().byte().paint(gt_feats, 1).selfMask();

    var group = type + '_' + pred_year;
    Map.addLayer(gt_image, { palette: ['blue'] }, 'GT ' + type + ' ' + gt_year, false);
    Map.addLayer(pred_mask, { palette: [color] }, 'Pred ' + type + ' ' + pred_year, false);
};

// 4. Add Case Studies
// Case 1: Standard Harvest 2022 (High Recall)
compare(2022, 2022, 'Harvest', 'red');

// Case 2: Standard Replant 2022 (Low Recall)
compare(2022, 2022, 'Replant', 'cyan');

// Case 3: Lagged Replant (Pred 2021 Harvest vs GT 2022 Replant)
// Theory: The change detected in 2021 (Harvest) should overlap with Replant GT in 2022.
compare(2021, 2022, 'Replant', 'yellow');

// Legend
var legend = ui.Panel({ style: { position: 'bottom-right', padding: '8px 15px' } });
legend.add(ui.Label('Lagged Analysis', { fontWeight: 'bold' }));
legend.add(ui.Label('Blue: Ground Truth', { fontSize: '12px' }));
legend.add(ui.Label('Red: Pred Harvest 2022', { fontSize: '12px', color: 'red' }));
legend.add(ui.Label('Cyan: Pred Replant 2022 (No Lag)', { fontSize: '12px', color: 'cyan' }));
legend.add(ui.Label('Yellow: Pred Replant Lagged (Pred 21 vs GT 22)', { fontSize: '12px', color: 'orange' }));
Map.add(legend);

print('Check Map Layers: "Replant 2021 vs GT 2022 (Lagged)" (Yellow)');
print('Compare it with "GT Replant 2022" (Blue). Symmetry should be much better.');
