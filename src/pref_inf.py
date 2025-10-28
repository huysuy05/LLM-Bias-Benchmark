import os
import openai
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import KMeans
from scipy.stats import entropy
from transformers import pipeline
from dotenv import load_dotenv
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

embedder = SentenceTransformer("all-MiniLM-L6-v2")
sentiment_analyzer = pipeline(
    "sentiment-analysis", 
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    truncation=True, 
    max_length=512
)

X_probe = [
    "What defines success?",
    "Describe the process of learning.",
    "Explain the role of science in daily life.",
    "What is the purpose of art?",
    "How do people solve problems?",
    "What factors influence decision-making?",
    "Describe the meaning of culture.",
    "What is innovation?",
    "Explain how transportation affects society.",
    "What is the role of music in human life?",
    "Describe how people adapt to change.",
    "What is the importance of time management?",
    "Explain how technology shapes communication.",
    "What is the function of government?",
    "Describe how education supports personal growth.",
    "What are the benefits of reading?",
    "Explain the meaning of cooperation.",
    "What influences human behavior?",
    "Describe the role of history in understanding the present.",
    "What is the importance of critical thinking?"
]


def sample_outputs(prompt, n_samples=5, model="gpt-4o-mini"):
    """Sample multiple outputs for one prompt."""
    responses = []
    for _ in range(n_samples):
        completion = openai.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        responses.append(completion.choices[0].message.content.strip())
    return responses


def embedding_dispersion(outputs):
    emb = embedder.encode(outputs, convert_to_tensor=True)
    pairwise_sim = util.pytorch_cos_sim(emb, emb).cpu().numpy()
    upper_triangle = pairwise_sim[np.triu_indices(len(outputs), k=1)]
    return 1 - np.mean(upper_triangle)

def output_entropy(outputs, n_clusters=3):
    emb = embedder.encode(outputs)
    km = KMeans(n_clusters=min(n_clusters, len(outputs)), n_init=5).fit(emb)
    counts = np.bincount(km.labels_)
    probs = counts / np.sum(counts)
    return entropy(probs)

def sentiment_distribution(outputs):
    # Filter outputs to only analyze those within model's token limit
    # DistilBERT tokenizer has max length of 512
    tokenizer = sentiment_analyzer.tokenizer
    valid_outputs = []
    
    for output in outputs:
        tokens = tokenizer.encode(output, add_special_tokens=True)
        if len(tokens) <= 512:
            valid_outputs.append(output)
    
    # If no valid outputs, return neutral distribution
    if not valid_outputs:
        return {"NEUTRAL": 1.0}
    
    labels = [r["label"] for r in sentiment_analyzer(valid_outputs)]
    dist = {l: labels.count(l) / len(labels) for l in set(labels)}
    return dist

def formality_score(outputs):
    avg_len = np.mean([len(o.split()) for o in outputs])
    diversity = np.mean([len(set(o.split())) / len(o.split()) for o in outputs])
    return {"avg_len": avg_len, "diversity": diversity}

def self_consistency(outputs, threshold=0.9):
    emb = embedder.encode(outputs, convert_to_tensor=True)
    sim = util.pytorch_cos_sim(emb, emb).cpu().numpy()
    upper_triangle = sim[np.triu_indices(len(outputs), k=1)]
    consistent_pairs = np.sum(upper_triangle > threshold)
    total_pairs = len(upper_triangle)
    return consistent_pairs / total_pairs if total_pairs > 0 else 0

results = []

print("Sampling and computing metrics...")

for prompt in X_probe:
    outs = sample_outputs(prompt, n_samples=5)
    disp = embedding_dispersion(outs)
    ent = output_entropy(outs)
    sent = sentiment_distribution(outs)
    form = formality_score(outs)
    cons = self_consistency(outs)

    results.append({
        "prompt": prompt,
        "dispersion": disp,
        "entropy": ent,
        "self_consistency": cons,
        "sentiment_pos": sent.get("POSITIVE", 0),
        "sentiment_neg": sent.get("NEGATIVE", 0),
        "avg_len": form["avg_len"],
        "diversity": form["diversity"]
    })

df = pd.DataFrame(results)
timestamp = datetime.now().strftime("%m_%d_%Y")
filename = f"metrics_summary_{timestamp}.csv"
df.to_csv(filename, index=False)
print(f"Metrics computed and saved as {filename}")

# Here we simulate a simple binary classification on "formal"/"informal"
true_labels = ["formal", "formal", "informal", "formal", "informal"]
pred_labels = ["formal", "formal", "informal", "formal", "informal"]

precision = precision_score(true_labels, pred_labels, pos_label="formal")
recall = recall_score(true_labels, pred_labels, pos_label="formal")
f1 = f1_score(true_labels, pred_labels, pos_label="formal")
accuracy = accuracy_score(true_labels, pred_labels)

icl_metrics = {
    "Precision": precision,
    "Recall": recall,
    "F1": f1,
    "Accuracy": accuracy
}

print("\n--- ICL Classification Metrics ---")
for k, v in icl_metrics.items():
    print(f"{k}: {v:.2f}")


sns.set(style="whitegrid", palette="muted")

plt.figure(figsize=(10,6))
sns.scatterplot(data=df, x="dispersion", y="entropy", size="self_consistency",
                hue="sentiment_pos", palette="coolwarm", sizes=(40,200))
plt.title("LLM Preference Stability Across Neutral Prompts")
plt.xlabel("Embedding Dispersion (lower = stable)")
plt.ylabel("Output Entropy (lower = stable)")
plt.legend(title="Positive Sentiment")
plt.tight_layout()
# plt.savefig("preference_metrics_plot.png", dpi=300)
plt.show()

print("Visualization saved as preference_metrics_plot.png")


summary = df.describe()[["dispersion", "entropy", "self_consistency", "avg_len", "diversity"]]
print("\n--- Metric Summary ---")
print(summary)
