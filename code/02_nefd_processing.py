import pandas as pd
import glob
import os

def process_nefd_data(nefd_dir="NEFD_statistics/Species_only/"):
    print("Processing NEFD Data...")
    
    # 1. File Mapping
    species_mapping = {
        "2019_2024_radiata_pine.csv": "R",
        "2019_2024_Douglas_fir.csv": "D",
        "2019_2024_eucalypt.csv": "Eu", 
        "2019_2024_cypress.csv": "Cy",
        "2019_2024_hardwoods.csv": "Oh", 
        "2019_2024_softwoods.csv": "Os",
    }
    
    age_mapping = {
        '1-5': 'S', '6-10': 'S', 
        '11-15': 'M', '16-20': 'M', 
        '21-25': 'M', '26-30': 'L', 
        '31-35': 'L', '36-40': 'L', '41-50': 'L', '51-60': 'L', '61-80': 'L',
    }
    nefd_age_cols = list(age_mapping.keys())
    
    # 2. Load and Aggregate
    all_species_dfs = []
    # Go up one level from 'code' to find 'NEFD_statistics' if running from 'code/'
    # Adjust path logic to be robust
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    full_nefd_dir = os.path.join(base_path, nefd_dir)
    
    if not os.path.exists(full_nefd_dir):
        # Fallback if running from root
        full_nefd_dir = os.path.abspath(nefd_dir)

    print(f"Looking for CSVs in: {full_nefd_dir}")
    nefd_files = glob.glob(os.path.join(full_nefd_dir, "2019_2024_*.csv"))
    
    if not nefd_files:
        print("No NEFD files found!")
        return None

    for f in nefd_files:
        file_name = os.path.basename(f)
        species_code = species_mapping.get(file_name)
        if species_code:
            df = pd.read_csv(f)
            df['species'] = species_code
            all_species_dfs.append(df)
            
    df_raw = pd.concat(all_species_dfs, ignore_index=True)
    
    # Melt to long format
    df_long = df_raw.melt(
        id_vars=["Year", "Territorial Authority", "species"], 
        value_vars=nefd_age_cols,
        var_name="nefd_age_class", 
        value_name="Area"
    )
    
    # Map to S/M/L groups
    df_long['age_group'] = df_long['nefd_age_class'].map(age_mapping)
    df_long = df_long.dropna(subset=['age_group'])
    
    # Aggregate
    df_agg = df_long.groupby(
        ["Year", "Territorial Authority", "species", "age_group"]
    ).Area.sum().reset_index()
    
    df_agg['state'] = df_agg['species'] + '_' + df_agg['age_group']
    df_agg = df_agg.drop(columns=['species', 'age_group'])
    
    # 3. Calculate Aging and Abrupt Changes
    AGING_S_YEARS = 10
    AGING_M_YEARS = 15 # Changed to 10 years (11-20)
    
    # Estimate Aging Out (Flow)
    df_agg['Aging_Out'] = 0.0
    df_agg.loc[df_agg['state'].str.endswith('_S'), 'Aging_Out'] = df_agg['Area'] / AGING_S_YEARS
    df_agg.loc[df_agg['state'].str.endswith('_M'), 'Aging_Out'] = df_agg['Area'] / AGING_M_YEARS
    
    # Sort for shifting
    df_agg = df_agg.sort_values(by=["Territorial Authority", "state", "Year"])
    
    # Calculate Net Change (Delta Area)
    df_agg['Area_prev_year'] = df_agg.groupby(["Territorial Authority", "state"]).Area.shift(1)
    df_agg['delta_Area'] = df_agg['Area'] - df_agg['Area_prev_year']
    
    # Pivot for calculation
    df_pivot = df_agg.pivot_table(
        index=["Territorial Authority", "Year"],
        columns="state",
        values=['Area', 'Aging_Out', 'delta_Area']
    )
    
    # Calculate delta_Abrupt
    # Logic: Abrupt Change = Net Change - (Inflow from younger) + (Outflow to older)
    # Note: The formula in the notebook was:
    # S: delta_S + Aging_Out_S (assuming 0 inflow from planting? No, planting IS the abrupt change)
    #    So Abrupt_S = (Area_t - Area_t-1) + Outflow_S - Inflow_0 (Inflow_0 is planting)
    #    Wait, if we want to FIND planting, Planting = (Area_t - Area_t-1) + Outflow_S
    #    Yes, that matches notebook: df_pivot[('delta_Abrupt', s_col)] = delta_s + df_pivot[('Aging_Out', s_col)]
    
    for species in set(species_mapping.values()):
        s_col = f'{species}_S'
        m_col = f'{species}_M'
        l_col = f'{species}_L'
        
        try:
            # Get columns
            delta_s = df_pivot[('delta_Area', s_col)]
            delta_m = df_pivot[('delta_Area', m_col)]
            delta_l = df_pivot[('delta_Area', l_col)]
            
            aging_out_s_curr = df_pivot[('Aging_Out', s_col)]
            aging_out_m_curr = df_pivot[('Aging_Out', m_col)]
            
            # Aging In comes from PREVIOUS year's Out of the younger class
            # We need to shift the Aging_Out columns by 1 year within each TA
            # Since pivot index is (TA, Year), we group by TA and shift
            
            aging_out_s_prev = df_pivot[('Aging_Out', s_col)].groupby(level='Territorial Authority').shift(1)
            aging_out_m_prev = df_pivot[('Aging_Out', m_col)].groupby(level='Territorial Authority').shift(1)
            
            # Calculate Abrupt
            # S: Planting = Net_S + Out_S (to M)
            df_pivot[('delta_Abrupt', s_col)] = delta_s + aging_out_s_curr
            
            # M: Net_M = In_S (from prev) - Out_M (to L) + Abrupt_M
            # -> Abrupt_M = Net_M - In_S + Out_M
            df_pivot[('delta_Abrupt', m_col)] = delta_m - aging_out_s_prev + aging_out_m_curr
            
            # L: Net_L = In_M (from prev) - Out_L (Harvest) + Abrupt_L (Other?)
            # Usually Harvest is negative Abrupt.
            # Abrupt_L = Net_L - In_M
            # (Notebook logic: delta_l - aging_out_m_prev)
            df_pivot[('delta_Abrupt', l_col)] = delta_l - aging_out_m_prev
            
        except KeyError:
            # Some species might not have all age classes in the data
            continue
            
    # Extract results
    df_abrupt = df_pivot['delta_Abrupt'].copy().reset_index()
    
    # Create year_period (e.g., 2019 -> 2019-2020 is wrong, data is Year end?)
    # Notebook logic: 'Year' 2020 means change from 2019 to 2020.
    # So Year 2020 row represents 2019-2020 period.
    # Let's verify: df_agg['Area_prev_year'] is shift(1).
    # If row is 2020, prev is 2019. delta is 2020 - 2019.
    # So period is 2019-2020.
    
    df_abrupt['year_period'] = df_abrupt['Year'].apply(lambda y: f"{y-1}-{y}")
    df_abrupt = df_abrupt.rename(columns={"Territorial Authority": "ta_name"})
    df_abrupt = df_abrupt.drop(columns=['Year'])
    
    # Rename columns to be clear
    df_abrupt.columns = [f'delta_Abrupt_{col}' if col not in ['ta_name', 'year_period'] else col for col in df_abrupt.columns]
    
    # Drop NaN (first year)
    df_abrupt = df_abrupt.dropna()
    
    print(f"Abrupt changes calculated. Shape: {df_abrupt.shape}")
    return df_abrupt

if __name__ == "__main__":
    df = process_nefd_data()
    if df is not None:
        output_path = "code/nefd_abrupt_changes.csv"
        df.to_csv(output_path, index=False)
        print(f"Saved to {output_path}")
