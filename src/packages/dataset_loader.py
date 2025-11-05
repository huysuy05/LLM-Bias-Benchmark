import pandas as pd
import json
from pathlib import Path

try:
    from textclass_benchmark import load_dataset as tc_load_dataset  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    tc_load_dataset = None

class DatasetLoader:
    def __init__(self, label_maps):
        self.label_maps = label_maps

    def _load_fixed_test_set(self, dataset_name, data_dir, rows_per_class=200):
        """
        Load a fixed test set if it exists, otherwise return None.
        Fixed test sets are created by src/create_fixed_test_set.py
        """
        data_path = Path(data_dir)
        fixed_test_file = data_path / f"test_fixed_{rows_per_class}per_class.jsonl"
        
        if not fixed_test_file.exists():
            return None
        
        print(f"[INFO] Loading fixed test set: {fixed_test_file}")
        
        records = []
        with open(fixed_test_file, 'r', encoding='utf-8') as f:
            for line in f:
                records.append(json.loads(line))
        
        df = pd.DataFrame(records)
        
        # Load metadata if available
        metadata_file = data_path / f"test_fixed_{rows_per_class}per_class.json"
        if metadata_file.exists():
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                print(f"[INFO] Fixed test set created: {metadata.get('created_at', 'unknown')}")
                print(f"[INFO] Total examples: {len(df)}, Rows per class: {rows_per_class}")
        
        return df

    def load_ag_news_data(self, data_dir="Data/ag_news", use_fixed_test=True, fixed_test_rows=200):
        label_map = self.label_maps['ag_news']
        
        # Try to load fixed test set first
        if use_fixed_test:
            fixed_df = self._load_fixed_test_set('ag_news', data_dir, fixed_test_rows)
            if fixed_df is not None:
                variants = {
                    "ag_news_balanced": fixed_df
                }
                return variants
        
        # Fall back to loading regular datasets
        # Load existing prepared files
        ag_news_imbalanced_data_99_to_1 = pd.read_parquet(f"{data_dir}/ag_news_train_imbalanced_99_to_1.parquet")
        balanced_data = pd.read_parquet(f"{data_dir}/ag_news_train_balanced.parquet")
        ag_news_imbalanced_data_49_to_1 = pd.read_parquet(f"{data_dir}/ag_news_train_imbalanced_49_to_1_ratio.parquet")
        
        # Map numeric labels to text labels
        balanced_data["label"] = balanced_data["label"].map(label_map)
        ag_news_imbalanced_data_99_to_1["label"] = ag_news_imbalanced_data_99_to_1["label"].map(label_map)
        ag_news_imbalanced_data_49_to_1["label"] = ag_news_imbalanced_data_49_to_1["label"].map(label_map)
        
        # Shuffle datasets
        ag_news_imbalanced_data_99_to_1 = ag_news_imbalanced_data_99_to_1.sample(frac=1).reset_index(drop=True)
        balanced_data = balanced_data.sample(frac=1).reset_index(drop=True)
        ag_news_imbalanced_data_49_to_1 = ag_news_imbalanced_data_49_to_1.sample(frac=1).reset_index(drop=True)
        
        # Create additional imbalanced datasets
        ag_news_world_majority_99 = self._split_ratio_for_ag_news(balanced_data, 'world', 200, 10)
        ag_news_sports_majority_99 = self._split_ratio_for_ag_news(balanced_data, 'sports', 200, 10)
        ag_news_business_majority_99 = self._split_ratio_for_ag_news(balanced_data, 'business', 490, 10)

        variants = {
            "ag_news_balanced": balanced_data
            # "ag_news_imbalanced_data_99_to_1": ag_news_imbalanced_data_99_to_1,
            # "ag_news_world_majority_99": ag_news_world_majority_99,
            # "ag_news_sports_majority_99": ag_news_sports_majority_99
        }

        tc_df = self._try_load_textclass_dataset("ag_news", label_map)
        if tc_df is not None:
            variants["ag_news_textclass_test"] = tc_df
        
        return variants
    
    def _split_ratio_for_ag_news(self, df, majority_label, majority_count, minority_count):
        parts = []
        labels = df['label'].unique().tolist()
        assert majority_label in labels, "Label not in provided labels"
        for lab in labels:
            if lab == majority_label:
                parts.append(df[df['label'] == lab].sample(majority_count, random_state=42))
            else:
                parts.append(df[df['label'] == lab].sample(minority_count, random_state=42))
        out = pd.concat(parts, ignore_index=True, sort=False)
        return out.sample(frac=1).reset_index(drop=True)

    def _try_load_textclass_dataset(self, dataset_name, label_map, split="test"):
        if tc_load_dataset is None:
            print(f"textclass_benchmark not installed; skipping TextClass Benchmark split for {dataset_name}.")
            return None

        try:
            dataset = tc_load_dataset(dataset_name, split=split)
        except Exception as exc:  # pragma: no cover - runtime guard
            print(f"Failed to load TextClass Benchmark split '{dataset_name}/{split}': {exc}")
            return None

        if hasattr(dataset, "to_pandas"):
            df = dataset.to_pandas()
        else:
            df = pd.DataFrame(dataset)
        if df.empty:
            print(f"TextClass Benchmark split '{dataset_name}/{split}' returned no rows.")
            return None

        if 'label' not in df.columns:
            for candidate in ['labels', 'target', 'y']:
                if candidate in df.columns:
                    df = df.rename(columns={candidate: 'label'})
                    break

        if 'text' not in df.columns:
            for candidate in ['sentence', 'content', 'document', 'review']:
                if candidate in df.columns:
                    df = df.rename(columns={candidate: 'text'})
                    break

        if 'text' not in df.columns or 'label' not in df.columns:
            print(f"TextClass Benchmark split '{dataset_name}/{split}' does not expose 'text'/'label' columns; skipping.")
            return None

        if df['label'].dtype != object:
            inverse_map = {idx: lab for idx, lab in label_map.items()}
            df['label'] = df['label'].map(inverse_map).fillna(df['label'])

        # Ensure ordering/shuffling matches expectation
        df = df[['text', 'label']].copy()
        df['label'] = df['label'].astype(str).str.lower()
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        return df
    

    def load_toxic_text_data(self, data_dir="Data/toxic_text"):
        """Load and prepare toxic text datasets."""
        toxic_label_map = self.label_maps['toxic_text']
        
        toxic_text = pd.read_csv(f"{data_dir}/train.csv")
        toxic_text = toxic_text[["comment_text", "toxic"]]
        toxic_text = toxic_text.rename(columns={"comment_text": "text", "toxic": "label"})
        toxic_text["label"] = toxic_text["label"].map(toxic_label_map)
        
        # Create different imbalanced datasets
        toxic_balanced = self._split_ratio_for_toxic_dataset(toxic_text, 'nontoxic', 250, 250)
        toxic_99_to_1 = self._split_ratio_for_toxic_dataset(toxic_text, 'nontoxic', 490, 10)
        toxic_49_to_1 = self._split_ratio_for_toxic_dataset(toxic_text, 'nontoxic', 450, 10)
        toxic_toxic_majority_99 = self._split_ratio_for_toxic_dataset(toxic_text, 'toxic', 490, 20)
        
        return {
            "toxic_text": toxic_balanced,
            "toxic_99_to_1": toxic_99_to_1
        }
    
    def _split_ratio_for_toxic_dataset(self, df, majority_label='nontoxic', majority_count=500, minority_count=20):
        parts = []
        for lab in df['label'].unique():
            if lab == majority_label:
                parts.append(df[df['label'] == lab].sample(majority_count, random_state=42))
            else:
                parts.append(df[df['label'] == lab].sample(minority_count, random_state=42))
        out = pd.concat(parts, ignore_index=True, sort=False)
        return out.sample(frac=1).reset_index(drop=True)
    

    def load_twitter_emotion_data(self, data_dir="Data/twitter_emotion"):
        """Load and prepare Twitter emotion datasets."""
        emotion_map = self.label_maps['twitter_emotion']
        
        emotion_df = pd.read_parquet(f"{data_dir}/twitter_emotion.parquet")
        emotion_df["label"] = emotion_df["label"].map(emotion_map)
        
        # Create different imbalanced datasets
        emotion_balanced = self._split_ratio_for_emotion_dataset(emotion_df, 'sadness', 100, 100)
        emotion_imbalanced_99_to_1 = self._split_ratio_for_emotion_dataset(emotion_df, 'sadness', 350, 10)
        emotion_imbalanced_49_to_1 = self._split_ratio_for_emotion_dataset(emotion_df, 'sadness', 100, 10)
        emotion_joy_majority_99 = self._split_ratio_for_emotion_dataset(emotion_df, 'joy', 350, 10)
        emotion_love_majority_99 = self._split_ratio_for_emotion_dataset(emotion_df, 'love', 350, 10)
        
        return {
            "emotion_df": emotion_balanced,
            "emotion_imbalanced_99_to_1": emotion_imbalanced_99_to_1,
            "emotion_love_majority_99": emotion_love_majority_99
        }
    
    def _split_ratio_for_emotion_dataset(self, df, majority_label='sadness', majority_count=200, minority_count=20):
        parts = []
        labels = df['label'].unique().tolist()
        for lab in labels:
            if lab == majority_label:
                parts.append(df[df['label'] == lab].sample(majority_count, random_state=42))
            else:
                parts.append(df[df['label'] == lab].sample(minority_count, random_state=42))
        out = pd.concat(parts, ignore_index=True, sort=False)
        return out.sample(frac=1).reset_index(drop=True)
    
    def load_mimic_data(self, data_dir="Data/MIMIC"):
        pass
    
    def reduce_size(self, dataset_dict, n_rows_per_class, random_state=42):
        """Return a copy of each dataset capped at *n_rows_per_class* per label."""
        if n_rows_per_class is None or n_rows_per_class <= 0:
            return dataset_dict

        reduced = {}
        for name, df in dataset_dict.items():
            samples = []
            for label, group in df.groupby("label"):
                k = min(len(group), n_rows_per_class)
                if k <= 0:
                    continue
                samples.append(group.sample(k, random_state=random_state))

            if samples:
                reduced_df = (
                    pd.concat(samples, ignore_index=True)
                    .sample(frac=1, random_state=random_state)
                    .reset_index(drop=True)
                )
                reduced[name] = reduced_df
            else:
                reduced[name] = df.copy()

        return reduced
            


