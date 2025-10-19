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
        ag_news_world_majority_99 = self._split_ratio_for_ag_news(balanced_data, 'world', 200, 10)
        ag_news_sports_majority_99 = self._split_ratio_for_ag_news(balanced_data, 'sports', 200, 10)
        ag_news_business_majority_99 = self._split_ratio_for_ag_news(balanced_data, 'business', 490, 10)
        
        return {
            "ag_news_balanced": balanced_data,
            "ag_news_imbalanced_data_99_to_1": ag_news_imbalanced_data_99_to_1,
            "ag_news_world_majority_99": ag_news_world_majority_99,
            "ag_news_sports_majority_99": ag_news_sports_majority_99
        }
    
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
    

    def load_twitter_emotion_data(self, data_dir="Data/twit"):
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
    
    def reduce_size(self, dataset_dict, n_rows_per_class):
        new_arr = []
        for name, df in dataset_dict.items():
            lab_map =  set(df["label"].unique())
            # print(lab_map)

            for lab in lab_map:
                # print(lab)
                class_samples = df[df["label"] == lab]
                size = min(len(class_samples), n_rows_per_class)
                new_arr.append(class_samples.sample(size, random_state=42))
            new_df = pd.concat(new_arr)
            dataset_dict[name] = new_df
            


ld = DatasetLoader({'ag_news': {0:'world',1:'sports',2:'business',3:'sci/tech'}})
ag_news = ld.load_ag_news_data()
print(ag_news)
# df = ld._split_ratio_for_ag_news

