"""
Create a fixed test set with 200 rows per class for consistent evaluation.
This ensures no randomness across runs and makes evaluation faster.
"""

import pandas as pd
import json
from pathlib import Path
import sys

# Add src to path
CURRENT_DIR = Path(__file__).resolve().parent
SRC_ROOT = CURRENT_DIR.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from packages.dataset_loader import DatasetLoader


# Label maps for supported datasets
LABEL_MAPS = {
    'ag_news': {
        0: 'world',
        1: 'sports',
        2: 'business',
        3: 'sci/tech'
    },
    'toxic_text': {
        0: 'nontoxic',
        1: 'toxic'
    },
    'twitter_emotion': {
        0: 'sadness',
        1: 'joy',
        2: 'love',
        3: 'anger',
        4: 'fear',
        5: 'surprise'
    }
}


def create_fixed_test_set(
    dataset_name: str,
    rows_per_class: int = 200,
    seed: int = 42,
    output_dir: Path = None
):
    """
    Create a fixed, balanced test set from a dataset.
    
    Args:
        dataset_name: Name of dataset (e.g., 'ag_news')
        rows_per_class: Number of rows per class
        seed: Random seed for reproducibility
        output_dir: Where to save the test set
    """
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent.parent / "Data" / dataset_name
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading {dataset_name} dataset...")
    dl = DatasetLoader(label_maps=LABEL_MAPS)
    
    # Load dataset using appropriate loader
    if dataset_name == 'ag_news':
        variants = dl.load_ag_news_data(use_fixed_test=False)  # Don't use fixed test when creating one!
    elif dataset_name == 'toxic_text':
        variants = dl.load_toxic_text_data()
    elif dataset_name == 'twitter_emotion':
        variants = dl.load_twitter_emotion_data()
    else:
        print(f"Error: Unsupported dataset '{dataset_name}'")
        return None
    
    label_map = LABEL_MAPS.get(dataset_name, {})
    
    # Use the balanced variant
    if dataset_name == 'ag_news':
        variant_key = 'ag_news_balanced'
    elif dataset_name == 'toxic_text':
        variant_key = 'toxic_text'
    elif dataset_name == 'twitter_emotion':
        variant_key = 'emotion_df'
    else:
        variant_key = f"{dataset_name}_balanced"
    
    if variant_key not in variants:
        print(f"Error: {variant_key} not found in variants")
        print(f"Available variants: {list(variants.keys())}")
        return None
    
    df = variants[variant_key]
    
    print(f"Original dataset size: {len(df)}")
    print(f"Label distribution:\n{df['label'].value_counts().sort_index()}")
    
    # Create balanced sample with fixed seed
    balanced_dfs = []
    for label in sorted(df['label'].unique()):
        label_df = df[df['label'] == label]
        
        # Sample with replacement if not enough examples
        n_available = len(label_df)
        if n_available < rows_per_class:
            print(f"Warning: Only {n_available} examples for '{label}', sampling with replacement")
            sampled = label_df.sample(n=rows_per_class, replace=True, random_state=seed)
        else:
            sampled = label_df.sample(n=rows_per_class, replace=False, random_state=seed)
        
        balanced_dfs.append(sampled)
        print(f"  {label}: sampled {len(sampled)} rows")
    
    # Combine and shuffle
    fixed_test_df = pd.concat(balanced_dfs, ignore_index=True)
    fixed_test_df = fixed_test_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    print(f"\nFixed test set created: {len(fixed_test_df)} total rows")
    print(f"Final label distribution:\n{fixed_test_df['label'].value_counts().sort_index()}")
    
    # Save as JSONL
    output_file = output_dir / f"test_fixed_{rows_per_class}per_class.jsonl"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for _, row in fixed_test_df.iterrows():
            record = {
                'text': row['text'],
                'label': row['label']
            }
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"\n✅ Fixed test set saved to: {output_file}")
    print(f"   Total examples: {len(fixed_test_df)}")
    print(f"   Rows per class: {rows_per_class}")
    print(f"   Random seed: {seed}")
    
    # Also save metadata
    metadata = {
        'dataset': dataset_name,
        'total_examples': len(fixed_test_df),
        'rows_per_class': rows_per_class,
        'seed': seed,
        'label_map': label_map,
        'label_counts': fixed_test_df['label'].value_counts().to_dict(),
        'created_at': pd.Timestamp.now().isoformat()
    }
    
    metadata_file = output_dir / f"test_fixed_{rows_per_class}per_class.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"   Metadata saved to: {metadata_file}")
    
    return output_file, metadata


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create fixed test set for consistent evaluation")
    parser.add_argument(
        '--dataset',
        type=str,
        default='ag_news',
        help='Dataset name (default: ag_news)'
    )
    parser.add_argument(
        '--rows-per-class',
        type=int,
        default=200,
        help='Number of rows per class (default: 200)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (default: Data/{dataset})'
    )
    
    args = parser.parse_args()
    
    create_fixed_test_set(
        dataset_name=args.dataset,
        rows_per_class=args.rows_per_class,
        seed=args.seed,
        output_dir=args.output_dir
    )
