"""
TODO:
1. Comment the metrics out, since we do not need to calculate metrics
2. Get multiple sampled output (y_1, y_2, etc) per input (x_i)
3. Count the number of labels for outputs

==> 2 main generations: 
    - We make the model generate its responses for n_samples
    - Then make the model classify its own generation
"""


import os
import openai
import numpy as np
import pandas as pd
from datetime import datetime
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import KMeans
from scipy.stats import entropy
from transformers import pipeline
from dotenv import load_dotenv
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from collections import Counter
import json

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# === Loading an Embedding Model === 
# embedder = SentenceTransformer("all-MiniLM-L6-v2")
# sentiment_analyzer = pipeline(
#     "sentiment-analysis", 
#     model="cardiffnlp/twitter-roberta-base-sentiment-latest",
#     truncation=True, 
#     max_length=512
# )

X_probe = [
    # Original prompts
    "What makes a daily routine effective?",
    "How do people organize their living spaces?",
    "What factors influence how people spend their free time?",
    "How do search engines organize information?",
    "What factors influence technology adoption?",
    "How do plants convert sunlight into energy?",
    "What makes ecosystems balanced?",
    "Describe the process of seasonal changes.",
    "Describe the process of cultural exchange.",
    "How do social norms develop over time?",
    "What factors contribute to community cohesion?",
    "How do organizations measure progress?",
    "What role does communication play in workplaces?",
    "How does physical activity benefit the body?",
    "What makes sleep restorative?",
    "What makes creative expressions meaningful?",
    "Describe the process of composing music.",
    "How do scientists test hypotheses?",
    "Describe the process of data analysis.",
    "How do people define personal values?",
    "What makes ideas persuasive?",

    # Additional challenging prompts
    "How do cognitive biases influence decision-making processes?",
    "What mechanisms underlie the formation of long-term memories?",
    "Describe the relationship between linguistic structures and thought patterns.",
    "How do complex systems maintain homeostasis amid external perturbations?",
    "What principles govern the emergence of collective intelligence in distributed networks?",
    "Explain the interplay between genetic predispositions and environmental factors in behavioral development.",
    "How do information cascades form in social networks and influence public opinion?",
    "Describe the cognitive processes involved in analogical reasoning and problem-solving.",
    "What factors determine the stability of social institutions across generational transitions?",
    "How do feedback mechanisms regulate biological systems at multiple scales?",
    "Explain the relationship between algorithmic complexity and computational efficiency.",
    "What processes enable the transmission and evolution of cultural memes?",
    "How do perceptual systems construct coherent representations from ambiguous sensory inputs?",
    "Describe the mechanisms of neuroplasticity and their role in learning adaptation.",
    "What principles govern the distribution of resources in self-organizing systems?",
    "How do conceptual frameworks shape the interpretation of empirical observations?",
    "Explain the dynamics of trust formation in decentralized networks.",
    "What factors influence the rate of technological convergence across different domains?",
    "How do hierarchical structures emerge in naturally occurring complex systems?",
    "Describe the relationship between entropy and information in physical systems.",
    "What processes underlie the development of expertise in complex domains?",
    "How do competing optimization criteria affect system design decisions?",
    "Explain the mechanisms of pattern recognition in both biological and artificial systems.",
    "What factors determine the robustness of ecological networks to external shocks?",
    "How do semantic networks organize conceptual knowledge and enable inference?",
    "Describe the interplay between individual agency and structural constraints in social systems.",
    "What principles govern the scaling relationships observed in biological and urban systems?",
    "How do confirmation biases affect the evaluation of scientific evidence?",
    "Explain the processes of cultural adaptation in response to environmental changes."
]
"""
NEW CODE: Run as a classification task using ICL, get multiple sampled outputs (y_1, y_2, ..., y_n) for each input x_i
"""

def create_icl_prompt(text_cl, examples, task_desc):
    prompt = f"""{task_desc}
        
        Examples:
"""
    for example in examples:
        prompt += f"Input: {example['input']}\nOutput: {example['output']}\n\n"

    prompt += f"Input: {text_cl}\nOutput: "
    return prompt

def classify_with_icl(text, model="gpt-4o-mini"):
    """Classify a text using ICL."""
    
    # Define your classification task and labels
    task_description = """Classify the writing style of the following text into one of these categories: 
                            - ANALYTICAL: Systematic, logical, fact-based explanation
                            - NARRATIVE: Storytelling, personal, descriptive  
                            - PRAGMATIC: Practical, actionable, advice-oriented
                            - PHILOSOPHICAL: Abstract, conceptual, meaning-seeking
                            - PERSUASIVE: Argumentative, convincing, opinion-based"""

    # Few-shot examples
    few_shot_examples = [
        {
            "input": "The scientific method involves forming hypotheses, conducting experiments, and analyzing results to draw conclusions based on empirical evidence.",
            "output": "ANALYTICAL"
        },
        {
            "input": "I remember when I first learned to ride a bike. The sun was shining, and my father ran beside me, holding the seat until I found my balance and rode off on my own.",
            "output": "NARRATIVE"
        },
        {
            "input": "To improve your productivity, start by prioritizing tasks, eliminating distractions, and using time-blocking techniques to focus on one thing at a time.",
            "output": "PRAGMATIC"
        },
        {
            "input": "The pursuit of happiness raises fundamental questions about human nature and whether fulfillment comes from within or through external achievements.",
            "output": "PHILOSOPHICAL"
        },
        {
            "input": "Everyone should adopt this approach because it's clearly the most efficient method that guarantees better results with less effort involved.",
            "output": "PERSUASIVE"
        }
    ]
    
    icl_prompt = create_icl_prompt(text, few_shot_examples, task_description)
    
    try:
        completion = openai.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": icl_prompt}],
            temperature=0.1,
            max_tokens=16
        )
        
        response = completion.choices[0].message.content.strip()
        # Clean up the response to get just the label
        for label in ["ANALYTICAL", "NARRATIVE", "PRAGMATIC", "PHILOSOPHICAL", "PERSUASIVE"]:
            if label in response:
                return label
        return response  # Return as-is if no exact match
    except Exception as e:
        print(f"Error classifying text: {e}")
        return "ERROR"


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

print("ICL...")
print("=" * 60)

results = []
all_classifications = []

for i, prompt in enumerate(X_probe):
    print(f"\n[{i+1}/{len(X_probe)}] Processing: '{prompt}'")

    # Get multiple sampled outputs per input
    outputs = sample_outputs(prompt, n_samples=5)

    prompt_clf = []
    for j, output in enumerate(outputs):
        if output:
            y_pred = classify_with_icl(output)
            prompt_clf.append(y_pred)
            print(f"Output: {i + 1}: {y_pred}")

            all_classifications.append({
                "prompt": prompt,
                "output_number": j + 1,
                "output_text": output,
                "classification": y_pred
            })
        else:
            prompt_clf.append("ERROR")
            print(f"    Output {j+1}: ERROR (empty output)")

    ct_label = Counter(prompt_clf)
    prompt_res = {
        "prompt": prompt,
        "total_samples": len(outputs),
        "label_counts": dict(ct_label),
        "classifications": prompt_clf
    }
    results.append(prompt_res)
    
    print(f"  Label counts: {dict(ct_label)}")

print("\n" + "=" * 60)
print("FINAL RESULTS: Label Counts Per Input")
print("=" * 60)

all_labels = []
for result in results:
    all_labels.extend(result['classifications'])

overall_counts = Counter(all_labels)
total_classifications = len(all_labels)

print(f"Total classifications: {total_classifications}")
print("Overall label distribution:")
for label, count in overall_counts.most_common():
    percentage = (count / total_classifications) * 100
    print(f"  {label}: {count} ({percentage:.1f}%)")

# Count prompts by dominant label
dominant_labels = []
for result in results:
    if result['label_counts']:
        dominant_label = max(result['label_counts'], key=result['label_counts'].get)
        dominant_labels.append(dominant_label)

dominant_counts = Counter(dominant_labels)
print(f"\nDominant labels per prompt ({len(dominant_labels)} prompts):")
for label, count in dominant_counts.most_common():
    percentage = (count / len(dominant_labels)) * 100
    print(f"  {label}: {count} ({percentage:.1f}%)")

# Save detailed results
timestamp = datetime.now().strftime("%m_%d_%Y_%H%M")

# Save per-prompt summary
summary_data = []
for result in results:
    row = {
        'prompt': result['prompt'],
        'total_samples': result['total_samples']
    }
    # Add counts for each label
    for label in ['ANALYTICAL', 'NARRATIVE', 'PRAGMATIC', 'PHILOSOPHICAL', 'PERSUASIVE', 'ERROR']:
        row[f'count_{label}'] = result['label_counts'].get(label, 0)
    summary_data.append(row)

summary_df = pd.DataFrame(summary_data)
summary_filename = f"label_counts_per_prompt_{timestamp}.csv"
summary_df.to_csv(summary_filename, index=False)

# Save all classifications
classifications_df = pd.DataFrame(all_classifications)
classifications_filename = f"all_classifications_{timestamp}.csv"
classifications_df.to_csv(classifications_filename, index=False)


print(f"\nResults saved:")
print(f"  - Per-prompt counts: {summary_filename}")
print(f"  - All classifications: {classifications_filename}")
 

# Display a sample of the data
print(f"\nSample of classified outputs:")
sample_df = classifications_df.head(3)
for _, row in sample_df.iterrows():
    print(f"  Prompt: {row['prompt'][:50]}...")
    print(f"  Output: {row['output_text'][:70]}...")
    print(f"  Classification: {row['classification']}\n")






"""
OLD CODE: With different metrics calculations (Semantic Dispersion, Entropy, etc) for semantic analysis
Commented out for potential future use
"""


# def embedding_dispersion(outputs):
#     emb = embedder.encode(outputs, convert_to_tensor=True)
#     pairwise_sim = util.pytorch_cos_sim(emb, emb).cpu().numpy()
#     upper_triangle = pairwise_sim[np.triu_indices(len(outputs), k=1)]
#     return 1 - np.mean(upper_triangle)

# def output_entropy(outputs, n_clusters=3):
#     emb = embedder.encode(outputs)
#     km = KMeans(n_clusters=min(n_clusters, len(outputs)), n_init=5).fit(emb)
#     counts = np.bincount(km.labels_)
#     probs = counts / np.sum(counts)
#     return entropy(probs)

# def sentiment_distribution(outputs):
#     # Filter outputs to only analyze those within model's token limit
#     # DistilBERT tokenizer has max length of 512
#     tokenizer = sentiment_analyzer.tokenizer
#     valid_outputs = []
    
#     for output in outputs:
#         tokens = tokenizer.encode(output, add_special_tokens=True)
#         if len(tokens) <= 512:
#             valid_outputs.append(output)
    
#     # If no valid outputs, return neutral distribution
#     if not valid_outputs:
#         return {"NEUTRAL": 1.0}
    
#     labels = [r["label"] for r in sentiment_analyzer(valid_outputs)]
#     dist = {l: labels.count(l) / len(labels) for l in set(labels)}
#     return dist

# def formality_score(outputs):
#     avg_len = np.mean([len(o.split()) for o in outputs])
#     diversity = np.mean([len(set(o.split())) / len(o.split()) for o in outputs])
#     return {"avg_len": avg_len, "diversity": diversity}

# def self_consistency(outputs, threshold=0.9):
#     emb = embedder.encode(outputs, convert_to_tensor=True)
#     sim = util.pytorch_cos_sim(emb, emb).cpu().numpy()
#     upper_triangle = sim[np.triu_indices(len(outputs), k=1)]
#     consistent_pairs = np.sum(upper_triangle > threshold)
#     total_pairs = len(upper_triangle)
#     return consistent_pairs / total_pairs if total_pairs > 0 else 0

# results = []

# print("Sampling and computing metrics...")

# for prompt in X_probe:
#     outs = sample_outputs(prompt, n_samples=5)
#     disp = embedding_dispersion(outs)
#     ent = output_entropy(outs)
#     sent = sentiment_distribution(outs)
#     form = formality_score(outs)
#     cons = self_consistency(outs)

#     results.append({
#         "prompt": prompt,
#         "dispersion": disp,
#         "entropy": ent,
#         "self_consistency": cons,
#         "sentiment_pos": sent.get("POSITIVE", 0),
#         "sentiment_neg": sent.get("NEGATIVE", 0),
#         "avg_len": form["avg_len"],
#         "diversity": form["diversity"]
#     })

# df = pd.DataFrame(results)
# timestamp = datetime.now().strftime("%m_%d_%Y")
# filename = f"metrics_summary_{timestamp}.csv"
# df.to_csv(filename, index=False)
# print(f"Metrics computed and saved as {filename}")

# # Here we simulate a simple binary classification on "formal"/"informal"
# true_labels = ["formal", "formal", "informal", "formal", "informal"]
# pred_labels = ["formal", "formal", "informal", "formal", "informal"]

# precision = precision_score(true_labels, pred_labels, pos_label="formal")
# recall = recall_score(true_labels, pred_labels, pos_label="formal")
# f1 = f1_score(true_labels, pred_labels, pos_label="formal")
# accuracy = accuracy_score(true_labels, pred_labels)

# icl_metrics = {
#     "Precision": precision,
#     "Recall": recall,
#     "F1": f1,
#     "Accuracy": accuracy
# }

# print("\n--- ICL Classification Metrics ---")
# for k, v in icl_metrics.items():
#     print(f"{k}: {v:.2f}")

# summary = df.describe()[["dispersion", "entropy", "self_consistency", "avg_len", "diversity"]]
# print("\n--- Metric Summary ---")
# print(summary)



