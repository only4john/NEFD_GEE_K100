# NERF-GEE Forest Activity Analysis (K=100 Optimization)

This repository contains the code and data used for analyzing forest harvesting and replanting activities in New Zealand using Google Earth Engine (GEE) annual satellite embeddings and NEFD (National Exotic Forest Description) statistics.

## Project Overview

The project employs a novel approach to correlate unsupervised spectral clusters (K=100) with ground-level forestry activities (Harvesting R_L and Replanting R_S). By using a **Genetic Algorithm (GA)**, we identify the specific combinations of clusters that best represent forestry changes recorded in statistical databases.

### Key Highlights:
- **K=100 Clustering**: Optimized clustering of AlphaEarth Annual Satellite Embeddings.
- **Genetic Algorithm (GA)**: Automated identifying of "cluster recipes" for R_L and R_S with cross-validation.
- **High Recall Validation**: Achieved **~89% spatial recall** for Radiata Harvesting (R_L) and **~78-90% spatial recall** for Radiata Replanting (R_S) using a lagged logic.
- **Lagged Replantation Logic**: Confirmed that change detected in Year T effectively predicts replanting recorded in Year T+1.

## Repository Structure

- `code/`: Core Python and JavaScript scripts.
    - `01_gee_clustering_export.py`: GEE script to cluster embeddings and export area tables per Territorial Authority (TA).
    - `02_nefd_processing.py`: Processes raw NEFD statistics into year-over-year abrupt change data.
    - `03_cluster_correlation.py`: The Genetic Algorithm engine that correlates GEE clusters with NEFD data.
    - `04_gee_validation.py`: Validation script to calculate spatial recall against high-resolution ground truth.
    - `05_gee_map_visualization.py`: Generates interactive Folium maps for visual inspection.
    - `GEE_Validation_Reproduction.js`: A standalone script for the GEE Code Editor to reproduce results visually.
- `data/`: Processed data and GEE exports.
    - `nefd_abrupt_changes.csv`: Cleaned statistical data for correlation.
    - `GEE_Exports_Universal_100/`: Area summaries of the 100 clusters across NZ districts.
- `results/`: Validation reports and trusted correlation recipes.
- `plots/`: Interactive HTML maps (Folium).

## Getting Started

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **GEE Authentication**:
   Ensure you have a Google Earth Engine account and have authenticated your environment:
   ```bash
   earthengine authenticate
   ```

3. **Running the Pipeline**:
   - Run `03_cluster_correlation.py` to identify the best cluster combinations.
   - Run `04_gee_validation.py` to verify the recall rates against ground truth.
   - Run `05_gee_map_visualization.py` to generate interactive maps.

## Results Summary (Gisborne District)

| Activity | Metric | Result | Note |
| :--- | :--- | :--- | :--- |
| **Radiata Harvest (R_L)** | Spatial Recall | **~89%** | Evaluated on 2020-2023 |
| **Radiata Replant (R_S)** | Lagged Recall | **~78-90%** | Pred(T-1) vs GT(T) |

---

# NERF-GEE 森林活动分析 (K=100 优化版)

本项目包含使用 Google Earth Engine (GEE) 年度卫星嵌入 (Embeddings) 与 NEFD (国家外来林描述) 统计数据分析新西兰森林采伐和补种活动的完整代码和数据。

## 项目简介

该项目采用了一种创新的方法，将无监督光谱聚类 (K=100) 与地面林业活动相关联。通过**遗传算法 (GA)**，我们识别出最能代表统计数据中记录的林业变化的特定聚类组合。

### 核心亮点：
- **K=100 聚类**: 基于 AlphaEarth 年度卫星嵌入的优化聚类。
- **遗传算法 (GA)**: 自动识别 R_L 和 R_S 的“聚类配方”，并经过交叉验证。
- **高召回率验证**: 辐射松采伐 (R_L) 的空间召回率达到 **~89%**；通过滞后逻辑，补种 (R_S) 的空间召回率达到 **~78-90%**。
- **滞后补种逻辑**: 证实了 T 年检测到的光谱变化能有效预测 T+1 年记录的补种活动。

## 联系方式与引用

如果你在研究中使用了本项目，请引用相关工作。

---
*Created for forestry validation and submission.*
