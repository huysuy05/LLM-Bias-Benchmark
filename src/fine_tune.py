import pandas as pd
import numpy as np
from huggingface_hub import login
from dotenv import load_dotenv
import os
import sys
import argparse
import subprocess
import random
import re
from pathlib import Path
from datetime import datetime
from time import time
import torch
from peft import LoraConfig
from functools import partial
import transformers
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, logging

# Load dataset
from dataset_loader import DatasetLoader


logging.set_verbosity_error()

class QwenFineTuning:
    """
    A class specifically written for fine-tuning Qwen2.5 models with 3 datasets in Data folder
    """
    def __init__(self, model_name, device=None):

        self.model_name = model_name
        self.device = self._setup_device(device)

        self.label_maps = {
            'ag_news': {
                0: "world",
                1: "sports", 
                2: "business",
                3: "sci/tech"
            },
            'toxic_text': {
                0: "nontoxic",
                1: "toxic"
            },
            'twitter_emotion': {
                0: "sadness",
                1: "joy",
                2: "love", 
                3: "anger",
                4: "fear",
                5: "surprise"
            }
        }

    def _setup_device(self, device):
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
                print("Using MPS device")
            elif torch.cuda.is_available():
                device = "cuda"
                print("Using CUDA device")
            else:
                device = "cpu"
                print("Using CPU device")
        else:
            print(f"Using specified device: {device}")
        return device

    
    def authenticate_hf(self, token=None):
        if token is None:
            # Try to get token from environment
            load_dotenv()
            token = os.getenv("hf_token")
            
        if token:
            login(token=token)
            print("Successfully authenticated with HuggingFace Hub")
        else:
            print("Warning: No HuggingFace token provided. Some models may not be accessible.")


    

    