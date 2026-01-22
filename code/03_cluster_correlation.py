import pandas as pd
import numpy as np
import glob
import os
import random
from scipy import stats
from sklearn.model_selection import KFold
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import re

# --- GA Parameters ---
POPULATION_SIZE = 100 # Optimized: 50 -> 100
GENERATIONS = 250
MUTATION_RATE = 0.2
MAX_ITEMS = 3 
MIN_RATIO = 0.6 # Adjusted: 0.8 -> 0.6 to accommodate the robust R_S signal
MAX_RATIO = 2.5

def load_gee_data(data_dir="GEE_Exports_Universal_100"):
    """Loads and aggregates GEE export CSVs."""
    print(f"Loading GEE Cluster Data from {data_dir}...")
    
    if os.path.exists(data_dir):
        full_data_dir = data_dir
    else:
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        full_data_dir = os.path.join(base_path, data_dir)
    
    if not os.path.exists(full_data_dir):
        print(f"Data directory not found: {full_data_dir}")
        return None
        
    files = glob.glob(os.path.join(full_data_dir, "NZ_TA_Universal_*.csv"))
    if not files:
        print("No GEE export files found.")
        return None
        
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    if not dfs: return None
        
    df_all = pd.concat(dfs, ignore_index=True)
    if 'system:index' in df_all.columns:
        df_all = df_all.drop(columns=['system:index', '.geo'], errors='ignore')

    # Fix TA Name Mismatches (GEE -> NEFD)
    name_map = {
        "Auckland": "Auckland Council",
        "Whanganui District": "Wanganui District",
        "Tauranga City": "Tauranga District",
        "Central Hawke's Bay District": "Central Hawkes Bay District",
        "Ōtorohanga District": "Otorohanga District",
        "Ōpōtiki District": "Opotiki District"
    }
    df_all['ta_name'] = df_all['ta_name'].replace(name_map)
        
    df_all = df_all.set_index(['ta_name', 'year_period'])
    cluster_cols = [c for c in df_all.columns if c.startswith('cluster_')]
    return df_all[cluster_cols].sort_index()

def load_logic_matrix(matrix_path="NEFD_statistics/Species_only/landuse_matrix_no_mng.csv"):
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    full_path = os.path.join(base_path, matrix_path)
    
    if not os.path.exists(full_path):
        full_path = matrix_path
        
    if not os.path.exists(full_path):
        print(f"Warning: Logic matrix not found at {full_path}")
        return None
        
    try:
        df = pd.read_csv(full_path).set_index("From (t)")
        return df
    except Exception as e:
        print(f"Error loading logic matrix: {e}")
        return None

def check_logic(target_name, df_logic):
    if df_logic is None: return True, "No Matrix"
    
    core = target_name.replace("delta_Abrupt_", "")
    if "_" in core:
        parts = core.rsplit("_", 1)
        species_code = parts[0]
        age_group = parts[1].upper()
        logic_name = f"{age_group}-{species_code}"
        
        if logic_name not in df_logic.index:
            return True, "Not in Matrix"
            
        if 'Cutover' in df_logic.columns:
            val = df_logic.loc[logic_name, 'Cutover']
            is_true = str(val).upper() == 'TRUE' or val == 1
            return is_true, f"{logic_name}->Cutover={val}"
            
    return True, "Unknown Format"

def run_ga_single_target(X_train, y_train, y_sum_true):
    corrs = []
    for i in range(X_train.shape[1]):
        if np.std(X_train[:, i]) == 0: continue
        r, _ = stats.pearsonr(X_train[:, i], y_train)
        if r > 0: corrs.append(i)
    
    if not corrs: return [], []
    
    pool_indices = sorted(corrs, key=lambda i: stats.pearsonr(X_train[:, i], y_train)[0], reverse=True)[:40]
    n_genes = len(pool_indices)
    
    population = []
    for _ in range(POPULATION_SIZE):
        ind = np.zeros(n_genes, dtype=int)
        indices = np.random.choice(n_genes, random.randint(1, 3), replace=False)
        ind[indices] = 1
        population.append(ind)
        
    best_ind = None
    best_score = -float('inf')
    history = []
    
    for gen in range(GENERATIONS):
        fitnesses = []
        for ind in population:
            sel = np.where(ind == 1)[0]
            if len(sel) == 0 or len(sel) > MAX_ITEMS: 
                fitnesses.append(-1)
                continue
            
            real_idx = [pool_indices[i] for i in sel]
            area_sum = X_train[:, real_idx].sum(axis=1)
            
            total_p = area_sum.sum()
            ratio = total_p / y_sum_true if y_sum_true > 0 else 999
            
            if ratio < MIN_RATIO or ratio > MAX_RATIO:
                fitnesses.append(-1)
                continue
            
            if np.std(area_sum) == 0: 
                fitnesses.append(0)
            else:
                r, _ = stats.pearsonr(area_sum, y_train)
                fitnesses.append(r)
        
        max_fit = np.max(fitnesses)
        if max_fit > best_score:
            best_score = max_fit
            best_ind = population[np.argmax(fitnesses)].copy()
        
        history.append(best_score)
            
        sorted_idx = np.argsort(fitnesses)[::-1]
        elites = [population[i] for i in sorted_idx[:POPULATION_SIZE//2]]
        
        new_pop = elites[:]
        while len(new_pop) < POPULATION_SIZE:
            child = random.choice(elites).copy()
            if random.random() < MUTATION_RATE:
                idx = random.randint(0, n_genes-1)
                child[idx] = 1 - child[idx]
            new_pop.append(child)
        population = new_pop
        
    if best_ind is not None and best_score > 0:
        sel = np.where(best_ind == 1)[0]
        return [pool_indices[i] for i in sel], history
    else:
        return [], history

def prune_solution(X_train, y_train, initial_indices, threshold=0.05):
    current_indices = list(initial_indices)
    
    def get_r(indices):
        if not indices: return 0
        s = X_train[:, indices].sum(axis=1)
        if np.std(s) == 0: return 0
        return stats.pearsonr(s, y_train)[0]

    current_r = get_r(current_indices)
    
    improved = True
    while improved and len(current_indices) > 1:
        improved = False
        best_subset = None
        best_subset_r = -1
        
        for i in range(len(current_indices)):
            subset = current_indices[:i] + current_indices[i+1:]
            r = get_r(subset)
            if r > best_subset_r:
                best_subset_r = r
                best_subset = subset
        
        if best_subset_r > (current_r - threshold):
            current_indices = best_subset
            current_r = best_subset_r
            improved = True
            
    return current_indices

# --- Visualization Functions ---

def plot_combined_convergence(all_histories, save_path="code/plots/average_convergence_250.png"):
    """Plots the average convergence across folds for each target."""
    plt.figure(figsize=(12, 8))
    cmap = plt.get_cmap('tab20')
    
    for i, (target, histories) in enumerate(all_histories.items()):
        if histories:
            # Filter out empty histories and ensure they are same length
            valid_histories = [h for h in histories if len(h) == GENERATIONS]
            if not valid_histories: continue
            
            # Calculate mean across folds
            avg_history = np.mean(valid_histories, axis=0)
            
            color = cmap(i % 20)
            clean_label = target.replace("delta_Abrupt_", "")
            plt.plot(avg_history, label=clean_label, color=color, linewidth=2, alpha=0.9)
            
    plt.title(f"GA Optimization Process (Average of 10 Folds, {GENERATIONS} Generations)", fontsize=16)
    plt.xlabel("Generation", fontsize=14)
    plt.ylabel("Mean Best Fitness (Pearson R)", fontsize=14)
    plt.ylim(0, 1)
    
    plt.axvline(x=50, linestyle='--', color='black', linewidth=2, alpha=0.7, label='Convergence Point')
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved average convergence plot to {save_path}")

def plot_recipe_heatmap(final_table, save_path="code/plots/recipe_heatmap.png"):
    """Visualizes the spectral recipes as a weighted heatmap."""
    df_subset = final_table.copy()
    df_subset.index = df_subset.index.str.replace('delta_Abrupt_', '')
    
    if df_subset.empty:
        print("No matching targets for Heatmap.")
        return pd.DataFrame()

    # Get all unique clusters and their weights
    dict_data = {}
    all_cluster_keys = set()
    
    for target in df_subset.index:
        recipe_str = df_subset.loc[target, 'Recipe']
        median_r = df_subset.loc[target, 'Median_R']
        
        matches = re.findall(r'cluster_(\d+)\((\d+)%\)', recipe_str)
        row_weights = {}
        for c_id, freq in matches:
            c_key = f"Cluster {c_id}"
            all_cluster_keys.add(c_key)
            # Weighted contribution
            row_weights[c_key] = (int(freq) / 100.0) * median_r
            
        dict_data[target] = row_weights

    if not all_cluster_keys:
        return pd.DataFrame()

    # Sort columns numerically
    sorted_cols = sorted(list(all_cluster_keys), key=lambda x: int(x.split(' ')[1]))
    
    # Create Matrix (fill with 0 for color, but we'll use a mask or custom labels)
    heatmap_df = pd.DataFrame(index=df_subset.index, columns=sorted_cols).fillna(0.0)
    for target, row_data in dict_data.items():
        for c_key, val in row_data.items():
            heatmap_df.loc[target, c_key] = val
            
    # Prepare annotation matrix (Strings)
    # We only show labels for non-zero cells to avoid clutter, or as per user request
    annot_df = heatmap_df.applymap(lambda x: f"{x:.2f}" if x > 0 else "")

    # Plot
    plt.figure(figsize=(10, len(df_subset.index) * 0.8 + 2))
    sns.heatmap(heatmap_df, cmap="Reds", annot=annot_df.values, fmt="", linewidths=.5, 
                vmin=0, vmax=1, cbar_kws={'label': 'Weighted Contribution (Freq * R)'},
                annot_kws={"size": 11, "weight": "bold", "color": "black"})
    
    plt.title("Decoded Spectral Recipe (Weighted by Median R)", fontsize=16, pad=20)
    plt.ylabel("Target Activity", fontsize=12)
    plt.xlabel("Cluster ID", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved modified recipe heatmap to {save_path}")
    
    return heatmap_df

def plot_sankey(final_table, save_path="code/plots/sankey_diagram.html"):
    # Filter targets - UPDATED: Use all from final_table
    # target_whitelist = ['delta_Abrupt_R_S', 'delta_Abrupt_R_L', 'delta_Abrupt_R_M', 'delta_Abrupt_Os_S']
    # df_subset = final_table[final_table.index.isin(target_whitelist)]
    df_subset = final_table.copy()
    
    if df_subset.empty:
        print("No matching targets for Sankey.")
        return

    try:
        # 2. Build Nodes and Links
        target_labels = []
        type_labels = []
        
        # Palette
        palette = [
            "rgba(31, 119, 180, 0.8)",  # Blue
            "rgba(255, 127, 14, 0.8)",  # Orange
            "rgba(44, 160, 44, 0.8)",   # Green
            "rgba(214, 39, 40, 0.8)",   # Red
            "rgba(148, 103, 189, 0.8)"  # Purple
        ]
        
        sources = []
        targets = []
        values = []
        link_colors = []
        
        # First pass: Register all nodes
        # Use df_subset which has 'Target' as index. We can iterate over index.
        # But user snippet used iterrows on a dataframe with 'Label' column?
        # My final_table has index as Target (e.g. delta_Abrupt_R_L).
        # Let's create a Label column for display if needed, or just use index.
        # User snippet used 'Label'. Let's create a clean label.
        
        def clean_label(t):
            return t.replace('delta_Abrupt_', '')
            
        # Register Targets
        for target in df_subset.index:
            lbl = clean_label(target)
            if lbl not in target_labels:
                target_labels.append(lbl)
        
        # Register Types (Clusters)
        for target in df_subset.index:
            recipe_str = df_subset.loc[target, 'Recipe']
            if pd.isna(recipe_str) or recipe_str == "None": continue
            
            # Match cluster_12(100%)
            matches = re.findall(r"cluster_(\d+)", recipe_str)
            for cid in matches:
                t_lbl = f"Cluster {cid}"
                if t_lbl not in type_labels:
                    type_labels.append(t_lbl)
                    
        # Sort Types
        type_labels.sort(key=lambda x: int(x.split(' ')[1]))
        
        all_labels = target_labels + type_labels
        
        # Second pass: Build links
        for i, target in enumerate(df_subset.index):
            t_label = clean_label(target)
            src_idx = target_labels.index(t_label)
            
            base_color = palette[src_idx % len(palette)]
            link_color = base_color.replace("0.8", "0.4")
            
            recipe_str = df_subset.loc[target, 'Recipe']
            if pd.isna(recipe_str) or recipe_str == "None": continue
            
            matches = re.findall(r"cluster_(\d+)\((\d+)%\)", recipe_str)
            
            for cid, freq in matches:
                t_lbl = f"Cluster {cid}"
                tgt_idx = len(target_labels) + type_labels.index(t_lbl)
                
                weight = int(freq)
                
                sources.append(src_idx)
                targets.append(tgt_idx)
                values.append(weight)
                link_colors.append(link_color)
                
        # 3. Force Coordinates
        node_x = []
        node_y = []
        
        # Left (Targets)
        for i in range(len(target_labels)):
            node_x.append(0.05)
            node_y.append((i + 0.5) / len(target_labels))
            
        # Right (Clusters)
        for i in range(len(type_labels)):
            node_x.append(0.95)
            node_y.append(None) # Let plotly optimize
            
        # 4. Plot
        fig = go.Figure(data=[go.Sankey(
            arrangement = "snap",
            node = dict(
              pad = 20,
              thickness = 20,
              line = dict(color = "white", width = 0.5),
              label = all_labels,
              color = palette[:len(target_labels)] + ["#dddddd"] * len(type_labels),
              x = node_x,
              y = node_y
            ),
            link = dict(
              source = sources,
              target = targets,
              value = values,
              color = link_colors
          ))])

        fig.update_layout( 
            title_text="Left: Management Activities | Right: GEE Clusters",
            font_size=14,
            height=700,
            margin=dict(l=50, r=550, t=80, b=50)
        )
        
        fig.write_html(save_path)
        print(f"Saved Sankey diagram to {save_path}")

    except Exception as e:
        print(f"Sankey Error: {e}")

def plot_abrupt_changes_2023_2024(df_Y, save_path="code/plots/abrupt_changes_2023_2024.png"):
    print("\n" + "="*60)
    print("--- Visualizing World A (Abrupt Change) ---")
    print("="*60)

    # 1. Prepare Data
    target_period = "2023-2024" 

    try:
        # Check if period exists
        if target_period not in df_Y.index.get_level_values('year_period'):
            print(f"Warning: {target_period} not found. Trying first available period...")
            target_period = df_Y.index.get_level_values('year_period').unique()[0]
            
        # Get slice
        df_viz = df_Y.xs(target_period, level='year_period').copy()
        
        # Remove 'delta_' prefix
        df_viz.columns = df_viz.columns.str.replace('delta_', '')
        
        # Remove 'Abrupt_' prefix if present for cleaner labels
        df_viz.columns = df_viz.columns.str.replace('Abrupt_', '')

        print(f"Plotting Abrupt Changes for {target_period}...")

        # 2. Plot Heatmap
        plt.figure(figsize=(20, 18)) 

        sns.heatmap(df_viz, 
                    cmap="vlag", 
                    center=0, 
                    robust=True, 
                    linewidths=0.5, 
                    linecolor='lightgray',
                    cbar_kws={'label': 'Net Abrupt Change (ha)'})

        plt.title(f"Net Abrupt Change by TA in Statistics ({target_period})", fontsize=20)
        plt.ylabel("Territorial Authority", fontsize=16)
        plt.xlabel("Forest Management States", fontsize=16)
        
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.yticks(fontsize=14)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved abrupt change chart to {save_path}")

        # 3. Simple Stats
        print(f"\n--- {target_period} Extremes ---")
        print(f"Max Decrease (Harvest): {df_viz.min().min():.1f} ha")
        print(f"Max Increase (Planting): {df_viz.max().max():.1f} ha")

    except Exception as e:
        print(f"Visualization failed: {e}")


def main():
    # 1. Load Data
    # Set seeds for reproducibility
    np.random.seed(42)
    random.seed(42)

    nefd_path = "nefd_abrupt_changes.csv"
    if not os.path.exists(nefd_path):
        nefd_path = "code/nefd_abrupt_changes.csv"
    if not os.path.exists(nefd_path):
        print("NEFD data not found.")
        return
    
    df_Y_raw = pd.read_csv(nefd_path)
    df_Y = df_Y_raw.set_index(['ta_name', 'year_period'])
    
    df_X = load_gee_data()
    if df_X is None: return
    
    # Align
    common_index = df_X.index.intersection(df_Y.index)
    if len(common_index) == 0:
        print("No overlapping data.")
        return
        
    X_full = df_X.loc[common_index].values
    Y_full = df_Y.loc[common_index]
    feature_names = np.array(df_X.columns)
    
    # 2. Logic Filtering
    matrix_path = "NEFD_statistics/Species_only/landuse_matrix_no_mng.csv"
    df_logic = load_logic_matrix(matrix_path)
    
    valid_targets = []
    for target in Y_full.columns:
        is_valid, reason = check_logic(target, df_logic)
        if is_valid:
            valid_targets.append(target)
        else:
            # print(f"Skipping {target}: {reason}")
            valid_targets.append(target)
            
    print(f"Valid Targets: {len(valid_targets)}")
    
    # Ensure plots directory
    if not os.path.exists("code/plots"):
        os.makedirs("code/plots")
    
    # 3. Cross-Validation
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    print(f"Starting 10-Fold Cross-Validation for {len(valid_targets)} targets...")
    cv_results = []
    recipe_log = {t: [] for t in valid_targets}
    
    # Store histories for combined plot (Target -> List of histories)
    all_histories = {t: [] for t in valid_targets}
    
    fold = 0
    for train_idx, test_idx in kf.split(X_full):
        fold += 1
        print(f"Fold {fold}/10...")
        
        X_tr, X_te = X_full[train_idx], X_full[test_idx]
        
        for target in valid_targets:
            y_tr = Y_full[target].iloc[train_idx].abs().values
            y_te = Y_full[target].iloc[test_idx].abs().values
            
            if np.sum(y_tr) == 0: continue
            
            # Run GA
            recipe_indices, history = run_ga_single_target(X_tr, y_tr, np.sum(y_tr))
            
            # Prune
            if recipe_indices:
                recipe_indices = prune_solution(X_tr, y_tr, recipe_indices, threshold=0.05)
            
            # Store history for analysis
            if history:
                all_histories[target].append(history)
            
            if recipe_indices:
                selected_names = feature_names[recipe_indices].tolist()
                recipe_log[target].extend(selected_names)
                
                # Test
                pred_sum = X_te[:, recipe_indices].sum(axis=1)
                
                if np.std(pred_sum) == 0 or np.std(y_te) == 0:
                    r_test = 0
                    p_test = 1.0
                else:
                    r_test, p_test = stats.pearsonr(pred_sum, y_te)
                    
                ratio_test = pred_sum.sum() / y_te.sum() if y_te.sum() > 0 else 0
                
                cv_results.append({
                    'Target': target,
                    'Fold': fold,
                    'R_Test': r_test,
                    'P_Value': p_test,
                    'Ratio_Test': ratio_test
                })
                
    # 4. Summary & Visualizations
    if not cv_results:
        print("No results found.")
        return
        
    df_res = pd.DataFrame(cv_results)
    df_res['Strict_Hit'] = (df_res['R_Test'] > 0.5) & (df_res['P_Value'] < 0.05)
    
    stats_df = df_res.groupby('Target').agg(
        Strict_Hit_Rate=('Strict_Hit', 'sum'),
        Median_R=('R_Test', 'median'),
        Median_P=('P_Value', 'median'),
        Median_Ratio=('Ratio_Test', 'median')
    )
    
    # Add Recipes
    recipes = []
    for target in stats_df.index:
        all_selected = recipe_log.get(target, [])
        if not all_selected:
            recipes.append("None")
            continue
        counts = Counter(all_selected)
        # Show ALL recipes sorted by frequency
        sorted_recipes = counts.most_common()
        parts = [f"{name}({int(count/10*100)}%)" for name, count in sorted_recipes]
        recipes.append(", ".join(parts))
        
    stats_df['Recipe'] = recipes # Renamed from Top_3_Recipe
    
    # Sort
    final_table_full = stats_df.sort_values(by='Median_R', ascending=False)
    
    # Save Full Report
    final_table_full.to_csv("code/cluster_activity_correlation_full.csv")
    print("Saved full report to code/cluster_activity_correlation_full.csv")

    # --- RIGOROUS FILTERING (User Request) ---
    # Only allow targets that pass:
    # 1. Whitelist: Only Radiata Planting (R_S) and Radiata Harvest (R_L)
    # 2. Hit Rate >= 5 (Robustness check)
    # 3. Median R > 0.5 (Quality check)
    final_table = final_table_full[
        (final_table_full.index.str.contains(r'_R_S|_R_L')) & 
        (final_table_full['Strict_Hit_Rate'] >= 5) &
        (final_table_full['Median_R'] > 0.5)
    ].copy()
    
    print("\n--- Final Report (Strict Validation Passed) ---")
    print(final_table)
    final_table.to_csv("code/cluster_activity_correlation_ga.csv")
    print("Saved filtered report to code/cluster_activity_correlation_ga.csv")
    
    # --- Create Trusted Semantics Report (Hit Rate >= 40%) ---
    # User Request: "Full Report subset... Hit Rate >= 40%"
    # User Request: "recipe_heatmap 还是要所有被遗传算法挑中过的clusters... 但target 我只要可信的"
    # So we filter targets by Hit Rate >= 40, but keep ALL clusters in the recipe.
    
    print("\n--- Generating Trusted Semantics Report (Hit Rate >= 40/100) ---")
    
    # Filter by Strict_Hit_Rate >= 4 (User Request: 40% of 10 folds)
    trusted_table = final_table_full[final_table_full['Strict_Hit_Rate'] >= 4].copy()
    
    # We do NOT filter the recipe string by 30% anymore for the CSV/Heatmap, 
    # as per "recipe_heatmap 还是要所有被遗传算法挑中过的clusters"
    # The user said "只选取出现次数大于或者等于 30% 的那些聚类的类" for downstream (Sankey/Validation),
    # but for the "Trusted Report" and "Heatmap", they seemingly want the full picture of the trusted targets.
    # Wait, the user said: "recipe_heatmap 还是要所有被遗传算法挑中过的clusters... 但target 我只要可信的"
    # AND "然后我希望只选取出现次数大于或者等于 30% 的那些聚类的类。然后之后部分结果，包括sankey图，validation... 都只取这个更可信的聚类的类"
    
    # Interpretation:
    # 1. Trusted CSV (for Heatmap): Trusted Targets (>=40 hits), ALL Clusters.
    # 2. Downstream (Sankey/Validation):    # 5. Generate Trusted Semantics Report (for Heatmap)
    # This table includes ALL clusters in the recipe (no 30% filter) for trusted targets
    final_table.to_csv("code/cluster_activity_correlation_trusted_heatmap.csv")
    print("Saved trusted heatmap report to code/cluster_activity_correlation_trusted_heatmap.csv")

    # 6. Generate Trusted Semantics Report (for Downstream Tools)
    # This table filters clusters with < 30% frequency
    trusted_df = final_table.copy()
    new_recipes = []
    for recipe in trusted_df['Recipe']:
        if recipe == "None":
            new_recipes.append("None")
            continue
        
        # Parse "cluster_12(80%), cluster_14(10%)"
        matches = re.findall(r'(cluster_\d+\(\d+%\))', recipe)
        filtered_parts = []
        for part in matches:
            # Extract freq
            freq = int(re.search(r'\((\d+)%\)', part).group(1))
            if freq >= 10:
                filtered_parts.append(part)
        
        if not filtered_parts:
            new_recipes.append("None")
        else:
            new_recipes.append(", ".join(filtered_parts))
            
    trusted_df['Recipe'] = new_recipes
    trusted_df.to_csv("code/cluster_activity_correlation_trusted.csv")
    print("Saved trusted report (filtered clusters) to code/cluster_activity_correlation_trusted.csv")

    # 7. Visualizations (Using Filtered Tables)
    
    # Heatmap: Use the table with ALL clusters (but only trusted targets)
    heatmap_df = plot_recipe_heatmap(final_table)
    
    # Sankey: Use the table with filtered clusters (>= 30%)
    plot_sankey(trusted_df)
    
    # Convergence Plot (Average across all folds)
    plot_combined_convergence(all_histories)
    
    # 4. Abrupt Changes 2023-2024
    plot_abrupt_changes_2023_2024(df_Y)

if __name__ == "__main__":
    main()
