import pandas as pd
from sklearn.metrics import classification_report
import numpy as np

def generate_true_report(predictions_file, ground_truth_file):
    """
    Loads, de-duplicates, cleans, and evaluates prediction and ground truth files, 
    and prints accurate classification reports.
    """
    try:
        predictions_df = pd.read_excel(predictions_file)
        ground_truth_df = pd.read_excel(ground_truth_file)
        print("Successfully loaded both Excel files.\n")
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return

    # --- NEW STEP: De-duplicate the data ---
    # Remove rows with duplicate email text, keeping the first instance.
    print(f"Original row count in ground truth: {len(ground_truth_df)}")
    ground_truth_df.drop_duplicates(subset=['text'], inplace=True)
    print(f"Row count after de-duplication: {len(ground_truth_df)}\n")

    print(f"Original row count in predictions: {len(predictions_df)}")
    predictions_df.drop_duplicates(subset=['text'], inplace=True)
    print(f"Row count after de-duplication: {len(predictions_df)}\n")
    
    # --- Data Preparation ---
    predictions_df = predictions_df[['text', 'pred_main', 'pred_sub']]
    ground_truth_df = ground_truth_df[['text', 'main_category', 'sub_category']].rename(columns={
        'main_category': 'true_main',
        'sub_category': 'true_sub'
    })

    # --- Merge and Clean ---
    merged_df = pd.merge(ground_truth_df, predictions_df, on='text', how='inner')
    valid_df = merged_df[~merged_df['pred_main'].isin(['api_error', 'parse_error', 'retry_failed'])].copy()

    # Standardize to lowercase
    for col in ['true_main', 'pred_main', 'true_sub', 'pred_sub']:
        valid_df[col] = valid_df[col].astype(str).str.lower().str.strip()

    # Replace any form of 'not applicable' or 'nan' with a standard value
    na_synonyms = ['nan', 'n/a', 'na', '']
    valid_df.replace(na_synonyms, 'not_applicable', inplace=True)
    valid_df.dropna(subset=['true_main', 'pred_main'], inplace=True)

    print(f"Total records matched and compared after cleaning and de-duplication: {len(valid_df)}\n")

    # --- Main Category Report ---
    print("--- Main Category Classification Report ---")
    main_labels = sorted(list(pd.unique(valid_df[['true_main', 'pred_main']].values.ravel('K'))))
    # Exclude 'not_applicable' from main report if it exists
    if 'not_applicable' in main_labels:
        main_labels.remove('not_applicable')

    main_report = classification_report(
        valid_df['true_main'], 
        valid_df['pred_main'], 
        labels=main_labels,
        zero_division=0
    )
    print(main_report)

    # --- Sub-Category Report (for 'recruiting' emails only) ---
    print("\n--- Sub-Category Classification Report (for 'recruiting' emails) ---")
    recruiting_df = valid_df[valid_df['true_main'] == 'recruiting']
    
    if not recruiting_df.empty:
        sub_labels = sorted(
            list(pd.unique(recruiting_df[['true_sub', 'pred_sub']].values.ravel('K')))
        )
        if 'not_applicable' in sub_labels:
            sub_labels.remove('not_applicable')

        sub_report = classification_report(
            recruiting_df['true_sub'], 
            recruiting_df['pred_sub'],
            labels=sub_labels,
            zero_division=0
        )
        print(sub_report)
    else:
        print("No 'recruiting' emails found to generate a sub-category report.")


if __name__ == "__main__":
    PREDICTIONS_FILENAME = "gemini_classified_output.xlsx"
    GROUND_TRUTH_FILENAME = "classified_data.xlsx"
    
    generate_true_report(PREDICTIONS_FILENAME, GROUND_TRUTH_FILENAME)