from huggingface_hub import HfApi
import os

"""
PUSH MODELS TO HUGGING FACE
"""

DATASET_NAME = "ag_news"
MODEL_NAME = f"Qwen/Qwen2.5-0.5B-Instruct"
PATH_TO_MODEL = f"finetuned_models/{DATASET_NAME}/{MODEL_NAME}_finetuned"
COMMIT_MSG = "Uploaded fine-tuned model to HGF"

api = HfApi(token=os.getenv("hf_token"))
api.upload_folder(
    folder_path=PATH_TO_MODEL,
    repo_id=f"HuyAugie/Qwen2.5-0.5B-{DATASET_NAME}",
    repo_type="model",
    commit_message=COMMIT_MSG
)
