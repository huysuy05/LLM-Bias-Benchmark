import pandas as pd
import json
import re
from pathlib import Path

try:
    from datasets import load_dataset  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    load_dataset = None

try:
    from textclass_benchmark import load_dataset as tc_load_dataset  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    tc_load_dataset = None

class DatasetLoader:
    def __init__(self, label_maps):
        self.label_maps = label_maps

    def _load_jsonl_records(self, file_path):
        records = []
        with open(file_path, 'r', encoding='utf-8') as handle:
            for line in handle:
                records.append(json.loads(line))
        return pd.DataFrame(records)

    def _load_fixed_test_set(self, dataset_name, data_dir, rows_per_class=200):
        """
        Load a fixed test set if it exists, otherwise return None.
        Fixed test sets are created by src/create_fixed_test_set.py
        """
        data_path = Path(data_dir)

        if isinstance(rows_per_class, (list, tuple)):
            candidates = list(rows_per_class)
        else:
            candidates = [rows_per_class]

        for rows in candidates:
            fixed_test_file = data_path / f"test_fixed_{rows}per_class.jsonl"
            if not fixed_test_file.exists():
                continue

            print(f"[INFO] Loading fixed test set: {fixed_test_file}")

            df = self._load_jsonl_records(fixed_test_file)

            metadata_file = data_path / f"test_fixed_{rows}per_class.json"
            if metadata_file.exists():
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    print(f"[INFO] Fixed test set created: {metadata.get('created_at', 'unknown')}")
                    print(f"[INFO] Total examples: {len(df)}, Rows per class: {metadata.get('rows_per_class', rows)}")

            df.attrs['rows_per_class'] = rows
            return df

        return None

    def _load_fixed_variant(self, data_dir: Path, filename: str):
        candidate = data_dir / filename
        if not candidate.exists():
            return None

        print(f"[INFO] Loading fixed variant: {candidate}")
        df = self._load_jsonl_records(candidate)
        meta_path = candidate.with_suffix('.json')
        if meta_path.exists():
            with open(meta_path, 'r', encoding='utf-8') as handle:
                metadata = json.load(handle)
                print(f"[INFO] Variant created: {metadata.get('created_at', 'unknown')}")
                print(f"[INFO] Total examples: {len(df)}")
        return df

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

    def load_ag_news_data(self, data_dir="Data/ag_news", use_fixed_test=True, fixed_test_rows=200):
        label_map = self.label_maps['ag_news']

        balanced_source = pd.read_parquet(f"{data_dir}/ag_news_train_balanced.parquet")
        balanced_source["label"] = balanced_source["label"].map(label_map)
        balanced_source = balanced_source.sample(frac=1, random_state=42).reset_index(drop=True)

        fixed_df = self._load_fixed_test_set('ag_news', data_dir, fixed_test_rows) if use_fixed_test else None
        if fixed_df is not None:
            balanced_variant = fixed_df.copy()
            base_for_variants = balanced_variant
        else:
            balanced_variant = self._balanced_sample(balanced_source, rows_per_class=fixed_test_rows, seed=42)
            base_for_variants = balanced_source

        variants = {
            "ag_news_balanced": balanced_variant
        }

        # Build additional 150:50 variants by rotating the majority class
        majority_variants = {}
        for majority_label in label_map.values():
            key = majority_label.replace('/', '_')
            fixed_variant_name = f"test_fixed_{key}_majority_150_to_50.jsonl"
            fixed_variant = self._load_fixed_variant(Path(data_dir), fixed_variant_name)
            if fixed_variant is None:
                fixed_variant = self._split_ratio_for_ag_news(
                    base_for_variants,
                    majority_label,
                    majority_count=150,
                    minority_count=50,
                )
            else:
                fixed_variant = self._standardise_variant_columns(fixed_variant)
            majority_variants[f"ag_news_{key}_majority_150_to_50"] = fixed_variant

        variants.update(majority_variants)

        # Load any additional fixed ratio variants that follow the naming convention
        ratio_pattern = re.compile(r"^test_fixed_(?P<label>.+)_majority_(?P<maj>\d+)_to_(?P<min>\d+)$")
        data_path = Path(data_dir)
        for path in data_path.glob("test_fixed_*_majority_*_to_*.jsonl"):
            match = ratio_pattern.match(path.stem)
            if not match:
                continue

            label_slug = match.group("label")
            maj_count = int(match.group("maj"))
            min_count = int(match.group("min"))

            if not (maj_count == 150 and min_count == 50):
                continue

            canonical_label = None
            for label in label_map.values():
                if label.replace('/', '_').lower() == label_slug.lower():
                    canonical_label = label
                    break

            if canonical_label is None:
                continue

            variant_key = f"ag_news_{label_slug}_majority_{maj_count}_to_{min_count}"
            if variant_key in variants:
                continue

            fixed_variant = self._load_fixed_variant(data_path, path.name)
            if fixed_variant is None:
                continue

            fixed_variant = self._standardise_variant_columns(fixed_variant)
            variants[variant_key] = fixed_variant

        tc_df = self._try_load_textclass_dataset("ag_news", label_map)
        if tc_df is not None:
            variants["ag_news_textclass_test"] = tc_df

        return variants

    @staticmethod
    def _standardise_variant_columns(df: pd.DataFrame) -> pd.DataFrame:
        columns = [col for col in df.columns if col in {"text", "label"}]
        if columns:
            df = df[columns].copy()
        if 'label' in df.columns:
            df['label'] = df['label'].astype(str).str.lower()
        if 'text' in df.columns:
            df['text'] = df['text'].astype(str)
        return df
    
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
        out = out.sample(frac=1, random_state=42).reset_index(drop=True)
        if {'text', 'label'}.issubset(out.columns):
            out = out[['text', 'label']].copy()
        if 'label' in out.columns:
            out['label'] = out['label'].astype(str).str.lower()
        return out

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
    

    def load_toxic_text_data(self, data_dir="Data/toxic_text", use_fixed_test=True, fixed_test_rows=200):
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
            "toxic_nontoxic_majority_150_to_50": self._split_ratio_for_toxic_dataset(
                toxic_text, 'nontoxic', majority_count=150, minority_count=50
            ),
            "toxic_toxic_majority_150_to_50": self._split_ratio_for_toxic_dataset(
                toxic_text, 'toxic', majority_count=150, minority_count=50
            ),
        }

        return variants
    
    def _split_ratio_for_toxic_dataset(self, df, majority_label='nontoxic', majority_count=150, minority_count=50):
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
        out = out.sample(frac=1, random_state=42).reset_index(drop=True)
        if {'text', 'label'}.issubset(out.columns):
            out = out[['text', 'label']].copy()
        if 'label' in out.columns:
            out['label'] = out['label'].astype(str).str.lower()
        return out
    
    def _split_ratio_generic(self, df, majority_label, majority_count, minority_count, seed=42):
        """Split into majority/minority with replacement as needed."""
        parts = []
        labels = df['label'].unique().tolist()
        for lab in labels:
            if lab == majority_label:
                parts.append(
                    df[df['label'] == lab].sample(
                        n=majority_count,
                        replace=len(df[df['label'] == lab]) < majority_count,
                        random_state=seed,
                    )
                )
            else:
                parts.append(
                    df[df['label'] == lab].sample(
                        n=minority_count,
                        replace=len(df[df['label'] == lab]) < minority_count,
                        random_state=seed,
                    )
                )
        out = pd.concat(parts, ignore_index=True, sort=False)
        out = out.sample(frac=1, random_state=seed).reset_index(drop=True)
        if {'text', 'label'}.issubset(out.columns):
            out = out[['text', 'label']].copy()
        if 'label' in out.columns:
            out['label'] = out['label'].astype(str).str.lower()
        return out


    def load_sst2_data(self, data_dir="Data/sst2", use_fixed_test=True, fixed_test_rows=200):
        """Load and prepare SST-2 sentiment dataset (Hugging Face if local file missing)."""
        label_map = self.label_maps['sst2']
        data_path = Path(data_dir) / "sst2.csv"

        if data_path.exists():
            df = pd.read_csv(data_path)
        else:
            if load_dataset is None:
                raise ImportError("datasets package required to load SST-2 from Hugging Face")
            ds = load_dataset("glue", "sst2", split="train")
            df = ds.to_pandas()

        if 'sentence' in df.columns:
            df = df.rename(columns={'sentence': 'text'})
        df = df[['text', 'label']].copy()
        df['label'] = df['label'].map(label_map)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        fixed_df = self._load_fixed_test_set('sst2', data_dir, fixed_test_rows) if use_fixed_test else None
        if fixed_df is not None:
            balanced_variant = fixed_df.copy()
        else:
            balanced_variant = self._balanced_sample(df, rows_per_class=fixed_test_rows, seed=42)

        variants = {"sst2": balanced_variant}
        for majority_label in label_map.values():
            slug = majority_label.replace('/', '_')
            variants[f"sst2_{slug}_majority_150_to_50"] = self._split_ratio_generic(
                df,
                majority_label=majority_label,
                majority_count=150,
                minority_count=50,
                seed=42,
            )

        return variants

    def load_hatexplain_data(self, data_dir="Data/hatexplain", use_fixed_test=True, fixed_test_rows=200):
        """Load and prepare HateXplain dataset (Hugging Face if local file missing)."""
        label_map = self.label_maps['hatexplain']
        data_path = Path(data_dir) / "hatexplain.jsonl"

        if data_path.exists():
            raw_df = pd.read_json(data_path, lines=True)
        else:
            if load_dataset is None:
                raise ImportError("datasets package required to load HateXplain from Hugging Face")
            ds = load_dataset("hatexplain", split="train", trust_remote_code=True)
            # Majority vote over annotator labels, join tokens to text
            records = []
            for ex in ds:
                labels = ex.get("annotators", {}).get("label", [])
                if labels:
                    counts = pd.Series(labels).value_counts()
                    label = int(counts.idxmax())
                else:
                    label = 2  # default to neutral if missing
                tokens = ex.get("post_tokens") or []
                text = " ".join(tokens)
                records.append({"text": text, "label": label})
            raw_df = pd.DataFrame(records)

        if 'text' not in raw_df.columns and 'content' in raw_df.columns:
            raw_df = raw_df.rename(columns={'content': 'text'})

        df = raw_df[['text', 'label']].copy()
        df['label'] = df['label'].map(label_map)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        fixed_df = self._load_fixed_test_set('hatexplain', data_dir, fixed_test_rows) if use_fixed_test else None
        if fixed_df is not None:
            balanced_variant = fixed_df.copy()
        else:
            balanced_variant = self._balanced_sample(df, rows_per_class=fixed_test_rows, seed=42)

        variants = {"hatexplain": balanced_variant}
        for majority_label in label_map.values():
            slug = majority_label.replace('/', '_')
            variants[f"hatexplain_{slug}_majority_150_to_50"] = self._split_ratio_generic(
                df,
                majority_label=majority_label,
                majority_count=150,
                minority_count=50,
                seed=42,
            )

        return variants
    
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
            
