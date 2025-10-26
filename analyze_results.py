
import pandas as pd
import glob
import os
import numpy as np

def analyze_all_results():
    """
    Analyzes all experiment result CSVs to find correlations with MCC score.
    """
    base_path = '/Volumes/huysuy05/Projects/Bias_of_LLMs/'
    results_dirs = [
        'results',
        'mlx_models_results',
    ]

    all_dfs = []
    for res_dir in results_dirs:
        full_path = os.path.join(base_path, res_dir)
        # Use recursive glob to find all csv files
        csv_files = glob.glob(os.path.join(full_path, '**', '*.csv'), recursive=True)
        for f in csv_files:
            # Exclude data files from analysis
            if 'Data/' in f:
                continue
            try:
                df = pd.read_csv(f)
                df['source_file'] = os.path.basename(f)
                all_dfs.append(df)
            except Exception as e:
                print(f"Could not read {f}: {e}")

    if not all_dfs:
        print("No results files found or read.")
        return

    combined_df = pd.concat(all_dfs, ignore_index=True)

    # --- Data Cleaning and Preparation ---
    combined_df.columns = combined_df.columns.str.lower().str.replace('.', '_', regex=False)
    
    # Fill NaN for prompting method specific columns
    combined_df['shots_minority'] = combined_df.get('shots_minority', pd.Series(np.nan)).fillna(0)
    combined_df['shots_majority'] = combined_df.get('shots_majority', pd.Series(np.nan)).fillna(0)
    combined_df['self_consistency_samples'] = combined_df.get('self_consistency_samples', pd.Series(np.nan)).fillna(0)

    # Determine prompting method
    combined_df['prompting_method'] = np.where(combined_df['self_consistency_samples'] > 0, 'SC', 'ICL')


    # --- Analysis ---
    print("--- MCC Performance Analysis ---")

    # 1. Best overall MCC
    best_mcc_row = combined_df.loc[combined_df['mcc'].idxmax()]
    print("\n--- Best Overall MCC Score ---")
    print(f"MCC: {best_mcc_row['mcc']:.4f}")
    print(f"  - Model: {best_mcc_row.get('model', 'N/A')}")
    print(f"  - Dataset: {best_mcc_row.get('dataset', 'N/A')}")
    print(f"  - Prompting: {best_mcc_row.get('prompting_method', 'N/A')}")
    if best_mcc_row.get('prompting_method') == 'ICL':
        print(f"  - Shots (Minority/Majority): {int(best_mcc_row.get('shots_minority', 0))}/{int(best_mcc_row.get('shots_majority', 0))}")
    else:
        print(f"  - SC Samples: {int(best_mcc_row.get('self_consistency_samples', 0))}")
    print(f"  - Dataset Ratio: {best_mcc_row.get('dataset_ratio', 'N/A')}")
    print(f"  - Source File: {best_mcc_row.get('source_file', 'N/A')}")

    # 2. MCC by Prompting Method
    print("\n--- MCC by Prompting Method (ICL vs. SC) ---")
    print(combined_df.groupby('prompting_method')['mcc'].agg(['mean', 'max', 'count']).sort_values(by='mean', ascending=False))

    # 3. MCC by Model
    print("\n--- MCC by Model ---")
    print(combined_df.groupby('model')['mcc'].agg(['mean', 'max', 'count']).sort_values(by='mean', ascending=False))

    # 4. MCC by Dataset
    print("\n--- MCC by Dataset ---")
    print(combined_df.groupby('dataset')['mcc'].agg(['mean', 'max', 'count']).sort_values(by='mean', ascending=False))

    # 5. MCC by Dataset Ratio
    print("\n--- MCC by Dataset Ratio ---")
    # Clean up ratio string for better grouping
    combined_df['ratio_cleaned'] = combined_df['dataset_ratio'].str.replace('_', ':').fillna('unknown')
    print(combined_df.groupby('ratio_cleaned')['mcc'].agg(['mean', 'max', 'count']).sort_values(by='mean', ascending=False))

    # 6. ICL: MCC by number of shots
    icl_df = combined_df[combined_df['prompting_method'] == 'ICL'].copy()
    if not icl_df.empty:
        icl_df['shots_total'] = icl_df['shots_minority'] + icl_df['shots_majority']
        print("\n--- ICL: MCC by Total Number of Shots ---")
        print(icl_df.groupby('shots_total')['mcc'].agg(['mean', 'max', 'count']).sort_values(by='mean', ascending=False))

    # 7. SC: MCC by number of samples
    sc_df = combined_df[combined_df['prompting_method'] == 'SC'].copy()
    if not sc_df.empty:
        print("\n--- SC: MCC by Number of Samples ---")
        print(sc_df.groupby('self_consistency_samples')['mcc'].agg(['mean', 'max', 'count']).sort_values(by='mean', ascending=False))


if __name__ == "__main__":
    analyze_all_results()
