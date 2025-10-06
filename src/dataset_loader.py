import pandas as pd

class DatasetLoader:
    def __init__(self, label_maps):
        self.label_maps = label_maps

    def load_ag_news_data(self, data_dir="Data/ag_news"):
        label_map = self.label_maps['ag_news']
        
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
        ag_news_world_majority_99 = self._split_ratio_for_ag_news(balanced_data, 'world', 980, 20)
        ag_news_sports_majority_99 = self._split_ratio_for_ag_news(balanced_data, 'sports', 980, 20)
        ag_news_business_majority_99 = self._split_ratio_for_ag_news(balanced_data, 'business', 980, 20)
        
        return {
            "ag_news_balanced": balanced_data,
            "ag_news_imbalanced_data_99_to_1": ag_news_imbalanced_data_99_to_1,
            "ag_news_imbalanced_data_49_to_1": ag_news_imbalanced_data_49_to_1,
            "ag_news_world_majority_99": ag_news_world_majority_99,
            "ag_news_sports_majority_99": ag_news_sports_majority_99,
            "ag_news_business_majority_99": ag_news_business_majority_99
        }
    
    def _split_ratio_for_ag_news(self, df, majority_label, majority_count, minority_count):
        parts = []
        labels = df['label'].unique().tolist()
        for lab in labels:
            if lab == majority_label:
                parts.append(df[df['label'] == lab].sample(majority_count, random_state=42))
            else:
                parts.append(df[df['label'] == lab].sample(minority_count, random_state=42))
        out = pd.concat(parts, ignore_index=True, sort=False)
        return out.sample(frac=1).reset_index(drop=True)
    

    def load_toxic_text_data(self, data_dir="Data/toxic_text"):
        """Load and prepare toxic text datasets."""
        toxic_label_map = self.label_maps['toxic_text']
        
        toxic_text = pd.read_csv(f"{data_dir}/train.csv")
        toxic_text = toxic_text[["comment_text", "toxic"]]
        toxic_text = toxic_text.rename(columns={"comment_text": "text", "toxic": "label"})
        toxic_text["label"] = toxic_text["label"].map(toxic_label_map)
        
        # Create different imbalanced datasets
        toxic_balanced = self._split_ratio_for_toxic_dataset(toxic_text, 'nontoxic', 500, 500)
        toxic_99_to_1 = self._split_ratio_for_toxic_dataset(toxic_text, 'nontoxic', 980, 20)
        toxic_49_to_1 = self._split_ratio_for_toxic_dataset(toxic_text, 'nontoxic', 940, 20)
        toxic_toxic_majority_99 = self._split_ratio_for_toxic_dataset(toxic_text, 'toxic', 980, 20)
        
        return {
            "toxic_text": toxic_balanced,
            "toxic_99_to_1": toxic_99_to_1,
            "toxic_49_to_1": toxic_49_to_1,
            "toxic_toxic_majority_99": toxic_toxic_majority_99
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
    

    def load_twitter_emotion_data(self, data_dir="Data/twit"):
        """Load and prepare Twitter emotion datasets."""
        emotion_map = self.label_maps['twitter_emotion']
        
        emotion_df = pd.read_parquet(f"{data_dir}/twitter_emotion.parquet")
        emotion_df["label"] = emotion_df["label"].map(emotion_map)
        
        # Create different imbalanced datasets
        emotion_balanced = self._split_ratio_for_emotion_dataset(emotion_df, 'sadness', 200, 200)
        emotion_imbalanced_99_to_1 = self._split_ratio_for_emotion_dataset(emotion_df, 'sadness', 950, 20)
        emotion_imbalanced_49_to_1 = self._split_ratio_for_emotion_dataset(emotion_df, 'sadness', 202, 20)
        emotion_joy_majority_99 = self._split_ratio_for_emotion_dataset(emotion_df, 'joy', 950, 20)
        emotion_love_majority_99 = self._split_ratio_for_emotion_dataset(emotion_df, 'love', 950, 20)
        
        return {
            "emotion_df": emotion_balanced,
            "emotion_imbalanced_99_to_1": emotion_imbalanced_99_to_1,
            "emotion_imbalanced_49_to_1": emotion_imbalanced_49_to_1,
            "emotion_joy_majority_99": emotion_joy_majority_99,
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

