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

    def _load_fixed_test_set(self, dataset_name, data_dir, rows_per_class=100):
        """
        Load a fixed test set if it exists, otherwise return None.
        Fixed test sets are created by src/create_fixed_test_set.py
        """
        data_path = Path(data_dir)

        if isinstance(rows_per_class, (list, tuple)):
            candidates = list(rows_per_class)
        else:
            candidates = [rows_per_class]

        # Always fall back to the legacy 200-row sets when present.
        if 200 not in candidates:
            candidates.append(200)

        for rows in candidates:
            fixed_test_file = data_path / f"test_fixed_{rows}per_class.jsonl"
            if not fixed_test_file.exists():
                continue

            print(f"[INFO] Loading fixed test set: {fixed_test_file}")

            records = []
            with open(fixed_test_file, 'r', encoding='utf-8') as f:
                for line in f:
                    records.append(json.loads(line))

            df = pd.DataFrame(records)

            metadata_file = data_path / f"test_fixed_{rows}per_class.json"
            if metadata_file.exists():
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    print(f"[INFO] Fixed test set created: {metadata.get('created_at', 'unknown')}")
                    print(f"[INFO] Total examples: {len(df)}, Rows per class: {metadata.get('rows_per_class', rows)}")

            df.attrs['rows_per_class'] = rows
            return df

        return None

    def _balanced_sample(self, df, rows_per_class, seed=42):
        """Return a balanced sample with *rows_per_class* rows per label."""
        if rows_per_class is None or rows_per_class <= 0:
            return df.sample(frac=1, random_state=seed).reset_index(drop=True)

        samples = []
        for label, group in df.groupby("label"):
            if group.empty:
                continue
            replace = len(group) < rows_per_class
            samples.append(
                group.sample(
                    n=rows_per_class,
                    replace=replace,
                    random_state=seed,
                )
            )

        if not samples:
            return df.copy()

        combined = (
            pd.concat(samples, ignore_index=True)
            .sample(frac=1, random_state=seed)
            .reset_index(drop=True)
        )
        return combined

    def load_ag_news_data(self, data_dir="Data/ag_news", use_fixed_test=True, fixed_test_rows=100):
        label_map = self.label_maps['ag_news']

        balanced_source = pd.read_parquet(f"{data_dir}/ag_news_train_balanced.parquet")
        balanced_source["label"] = balanced_source["label"].map(label_map)
        balanced_source = balanced_source.sample(frac=1, random_state=42).reset_index(drop=True)

        fixed_df = self._load_fixed_test_set('ag_news', data_dir, fixed_test_rows) if use_fixed_test else None
        if fixed_df is not None:
            balanced_variant = fixed_df.copy()
        else:
            balanced_variant = self._balanced_sample(balanced_source, rows_per_class=fixed_test_rows, seed=42)

        variants = {
            "ag_news_balanced": balanced_variant
        }

        # Build additional 99:1 variants by rotating the majority class
        majority_variants = {}
        for majority_label in label_map.values():
            key = majority_label.replace('/', '_')
            majority_variants[f"ag_news_{key}_majority_99_to_1"] = self._split_ratio_for_ag_news(
                balanced_source,
                majority_label,
                majority_count=99,
                minority_count=1,
            )

        variants.update(majority_variants)

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
                parts.append(
                    df[df['label'] == lab].sample(
                        n=majority_count,
                        replace=len(df[df['label'] == lab]) < majority_count,
                        random_state=42,
                    )
                )
            else:
                parts.append(
                    df[df['label'] == lab].sample(
                        n=minority_count,
                        replace=len(df[df['label'] == lab]) < minority_count,
                        random_state=42,
                    )
                )
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
    

    def load_toxic_text_data(self, data_dir="Data/toxic_text", use_fixed_test=True, fixed_test_rows=100):
        """Load and prepare toxic text datasets."""
        toxic_label_map = self.label_maps['toxic_text']

        toxic_text = pd.read_csv(f"{data_dir}/train.csv")
        toxic_text = toxic_text[["comment_text", "toxic"]]
        toxic_text = toxic_text.rename(columns={"comment_text": "text", "toxic": "label"})
        toxic_text["label"] = toxic_text["label"].map(toxic_label_map)
        toxic_text = toxic_text.sample(frac=1, random_state=42).reset_index(drop=True)

        fixed_df = self._load_fixed_test_set('toxic_text', data_dir, fixed_test_rows) if use_fixed_test else None
        if fixed_df is not None:
            balanced_variant = fixed_df.copy()
        else:
            balanced_variant = self._balanced_sample(toxic_text, rows_per_class=fixed_test_rows, seed=42)

        variants = {
            "toxic_text": balanced_variant,
            "toxic_nontoxic_majority_99_to_1": self._split_ratio_for_toxic_dataset(
                toxic_text, 'nontoxic', majority_count=99, minority_count=1
            ),
            "toxic_toxic_majority_99_to_1": self._split_ratio_for_toxic_dataset(
                toxic_text, 'toxic', majority_count=99, minority_count=1
            ),
        }

        return variants
    
    def _split_ratio_for_toxic_dataset(self, df, majority_label='nontoxic', majority_count=500, minority_count=20):
        parts = []
        for lab in df['label'].unique():
            if lab == majority_label:
                parts.append(
                    df[df['label'] == lab].sample(
                        n=majority_count,
                        replace=len(df[df['label'] == lab]) < majority_count,
                        random_state=42,
                    )
                )
            else:
                parts.append(
                    df[df['label'] == lab].sample(
                        n=minority_count,
                        replace=len(df[df['label'] == lab]) < minority_count,
                        random_state=42,
                    )
                )
        out = pd.concat(parts, ignore_index=True, sort=False)
        return out.sample(frac=1).reset_index(drop=True)
    

    def load_twitter_emotion_data(self, data_dir="Data/twitter_emotion", use_fixed_test=True, fixed_test_rows=100):
        """Load and prepare Twitter emotion datasets."""
        emotion_map = self.label_maps['twitter_emotion']

        emotion_df = pd.read_parquet(f"{data_dir}/twitter_emotion.parquet")
        emotion_df["label"] = emotion_df["label"].map(emotion_map)
        emotion_df = emotion_df.sample(frac=1, random_state=42).reset_index(drop=True)

        fixed_df = self._load_fixed_test_set('twitter_emotion', data_dir, fixed_test_rows) if use_fixed_test else None
        if fixed_df is not None:
            balanced_variant = fixed_df.copy()
        else:
            balanced_variant = self._balanced_sample(emotion_df, rows_per_class=fixed_test_rows, seed=42)

        variants = {
            "emotion_df": balanced_variant,
        }

        for majority_label in emotion_map.values():
            slug = majority_label.replace('/', '_')
            variants[f"emotion_{slug}_majority_99_to_1"] = self._split_ratio_for_emotion_dataset(
                emotion_df,
                majority_label=majority_label,
                majority_count=99,
                minority_count=1,
            )

        return variants
    
    def _split_ratio_for_emotion_dataset(self, df, majority_label='sadness', majority_count=200, minority_count=20):
        parts = []
        labels = df['label'].unique().tolist()
        for lab in labels:
            if lab == majority_label:
                parts.append(
                    df[df['label'] == lab].sample(
                        n=majority_count,
                        replace=len(df[df['label'] == lab]) < majority_count,
                        random_state=42,
                    )
                )
            else:
                parts.append(
                    df[df['label'] == lab].sample(
                        n=minority_count,
                        replace=len(df[df['label'] == lab]) < minority_count,
                        random_state=42,
                    )
                )
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
            


