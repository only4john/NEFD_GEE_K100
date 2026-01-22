# Project Parameters

## 1. GEE Clustering & Export (`01_gee_clustering_export.py`)
| Parameter | Value | Description |
| :--- | :--- | :--- |
| `NUMBER_OF_CLUSTERS` | **100** | Number of K-Means clusters (K). |
| `SIMILARITY_THRESHOLD` | **0.95** | Cosine similarity threshold to filter non-changes. |
| `START_YEAR` | **2019** | Start year for analysis. |
| `END_YEAR` | **2024** | End year for analysis. |

## 2. NEFD Data Processing (`02_nefd_processing.py`)
| Parameter | Value | Description |
| :--- | :--- | :--- |
| `MATURE_AGE_MIN` | **11** | Minimum age for "Mature" forest. |
| `MATURE_AGE_MAX` | **25** | Maximum age for "Mature" forest. |
| `AGING_PERIOD` | **15** | Years to look back for "Aging" calculation. |

## 3. Correlation Analysis (GA) (`03_cluster_correlation.py`)
| Parameter | Value | Description |
| :--- | :--- | :--- |
| `GENERATIONS` | **150** | Number of generations to evolve. |
| `MUTATION_RATE` | **0.08** | Probability of gene mutation. |
| `MAX_ITEMS` | **5** | Maximum number of clusters allowed in a recipe. |
| `MIN_RATIO` | **0.8** | Minimum allowed Area_Predicted / Area_Actual ratio. |
| `MAX_RATIO` | **5.0** | Maximum allowed Area_Predicted / Area_Actual ratio. |
| `CV_FOLDS` | **10** | Number of folds for Cross-Validation. |
| `HIT_RATE_THRESHOLD` | **4** | Minimum successful folds (out of 10) to include in final report. |

## 4. Validation (`04_gee_validation.py`)
| Parameter | Value | Description |
| :--- | :--- | :--- |
| `YEARS_TO_CALCULATE` | `[2020, 2021, 2022, 2023]` | Years to run spatial validation on. |
| `TARGET_TA_NAME` | `'Gisborne District'` | Region used for validation. |
| `VALIDATION_ASSET` | `.../East_Coast_Small_scale_forests_2025` | Ground truth dataset path. |

## 5. Visualization (`05_gee_map_visualization.py`)
| Parameter | Value | Description |
| :--- | :--- | :--- |
| `VISUALIZATION_YEAR` | **2023** | Year to visualize on the map. |
| `MAP_CENTER` | `[-38.6625, 177.98]` | Center coordinates (Gisborne). |

## 6. LCDB Masking Codes
The analysis masks pixels based on the Land Cover Database (LCDB) to focus on relevant forestry areas.
- **71**: Exotic Forest
- **64**: Forest - Harvested
- **40**: High Producing Exotic Grassland
- **41**: Low Producing Grassland
- **50**: Fernland
- **52**: Manuka and/or Kanuka
