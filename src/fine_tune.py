# import pandas as pd
# import numpy as np
from huggingface_hub import login
from dotenv import load_dotenv
import os
import argparse
import random
from pathlib import Path
from datetime import datetime
from time import time
import torch
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from functools import partial
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, logging, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
import json

# Load dataset
from dataset_loader import DatasetLoader


logging.set_verbosity_error()

class QwenFineTuning:
    """
    A class specifically written for fine-tuning Qwen2.5 models with 3 datasets in Data folder
    """
    def __init__(self, model_name, output_dir, device=None):

        self.model_name = model_name
        self.device = self._setup_device(device)
        self.output_dir = output_dir

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

    
    def load_model_tokenizer(self, bnb_config):
        """
        Load model and tokenizer using the appropriate constructors.

        Uses `from_pretrained` to avoid direct constructor misuse. Supports bitsandbytes
        quantized loading when `bnb_config` is provided.
        """
        model = None
        try:
            if bnb_config:
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config=bnb_config,
                    device_map="auto",
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map="auto")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load model '{self.model_name}' (bnb_config={bool(bnb_config)}): {e}"
            ) from e

        tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_auth_token=True)
        if getattr(tokenizer, "pad_token", None) is None:
            tokenizer.pad_token = tokenizer.eos_token

        return model, tokenizer

    def build_prompt(self, df, text, label_map, shots_minority=0, shots_majority=0):
        """
        Build prompt using `shots_majority` for the inferred majority label (from df)
        and `shots_minority` for the other labels.
        """
        assert shots_minority is not None and shots_majority is not None, (
            "Please provide 'shots_minority' and 'shots_majority' parameters"
        )

        labels = list(label_map.values())
        prompt = (
            f"You are a powerful, precise, and helpful assistant that classifies text into well-defined categories, NO MATTER THE CONTEXT."
            f" IMPORTANT: CHOOSE ONE WORD FROM THESE CATEGORIES: {', '.join(labels)}."
            f" Respond with exactly one word: the single best category inside the given categories, DO NOT ANSWER ANY OTHER CATEGORIES BESIDES THE GIVEN ONE."
            f" Do not explain your choice, provide reasoning, or output anything else."
        )

        # Infer majority label from df (if possible)
        maj_label = None
        try:
            counts = df['label'].value_counts()
            if len(counts) > 0:
                maj_label = counts.idxmax()
        except Exception:
            maj_label = None

        # Collect few-shot examples per label according to inferred majority/minority shots
        few_shots_example = []
        for lab in labels:
            # If we can't infer majority, treat all labels equally (use shots_minority for all)
            if maj_label is not None and lab == maj_label:
                n = int(shots_majority)
            else:
                n = int(shots_minority)

            if n <= 0:
                continue

            avail = df[df['label'] == lab]
            k = min(n, len(avail))
            if k <= 0:
                continue

            samples = avail.sample(k, random_state=42)
            for _, r in samples.iterrows():
                few_shots_example.append({'text': r['text'], 'label': r['label']})

        if few_shots_example:
            random.shuffle(few_shots_example)
            prompt += "\n\nLearn from these examples to understand context and edge cases:\n\n"
            for ex in few_shots_example:
                prompt += f"Review: \"{ex['text']}\"\nCategory: {ex['label']}\n\n"

        prompt += f"Review: \"{text}\"\nCategory:"
        return prompt

    def preprocess_dataset(self, tokenizer, seed, df, shots, max_length):
        # Turn pandas df into hugging face format to fine-tune
        dataset_hgf = Dataset.from_pandas(df)

        dataset_hgf["prompted_text"] = dataset_hgf.apply(
            lambda row: self.build_prompt(
                df = dataset_hgf,
                shots = shots
            ) + f"\nText: {row['text']}\nCategory:",
            axis=1
        )

        def tokenize_batch(batch, tokenizer, max_length):
            return tokenizer(
                batch["prompted_text"],
                padding="max_length",
                truncation=True,
                max_length=max_length
            )

        _preprocessing_function = partial(tokenize_batch, tokenizer=tokenizer, max_length=max_length)
        dataset_hgf = dataset_hgf.map(
            _preprocessing_function,
            batched=True
        )

        dataset_hgf = dataset_hgf.shuffle(seed=seed)

        return dataset_hgf
    

    def get_qlora_config(self, load_in_4bit,bnb_4bit_use_double_quant,bnb_4bit_quant_type,bnb_4bit_compute_dtype,r,lora_alpha,target_modules,lora_dropout, bias, task_type):
        bnb_config = BitsAndBytesConfig(
        load_in_4bit = load_in_4bit,
        bnb_4bit_use_double_quant = bnb_4bit_use_double_quant,
        bnb_4bit_quant_type = bnb_4bit_quant_type,
        bnb_4bit_compute_dtype = bnb_4bit_compute_dtype,
    )

        lora_config = LoraConfig(
            r = r,
            lora_alpha = lora_alpha,
            target_modules = target_modules,
            lora_dropout = lora_dropout,
            bias = bias,
            task_type = task_type,
        )

        return bnb_config, lora_config
    
    def find_all_linear_names(self, model):
        linear_module_names = []
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                linear_module_names.append(name)
        return linear_module_names
    
    def create_peft_config(self, lora_r, lora_alpha, target_modules, lora_dropout, bias, task_type):
        """
        Create a LoRA PEFT configuration.
        """
        return LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias=bias,
            task_type=TaskType[task_type.upper()]
        )

    def prepare_model_for_fine_tune(self, model: AutoModelForCausalLM,
                                    lora_r: int,
                                    lora_alpha: int,
                                    lora_dropout: float,
                                    bias: str,
                                    task_type: str) -> AutoModelForCausalLM:
        """
        Prepares the model for fine-tuning.

        Args:
            model (AutoModelForCausalLM): The model that will be fine-tuned.
            lora_r (int): Lora attention dimension.
            lora_alpha (int): The alpha parameter for Lora scaling.
            lora_dropout (float): The dropout probability for Lora layers.
            Bias type for Lora. Can be 'none', 'all' or 'lora_only'. If 'all' or 'lora_only', the
                corresponding biases will be updated during training. Be aware that this means that, even when disabling
                the adapters, the model will not produce the same output as the base model would have without adaptation.
            task_type (str): The task type for the model.

        Returns:
            AutoModelForCausalLM: The model prepared for fine-tuning.
        """

        model.gradient_checkpointing_enable()


        model = prepare_model_for_kbit_training(model)


        target_modules = self.find_all_linear_names(model)

        # Create PEFT configuration for these modules and wrap the model to PEFT
        peft_config = self.create_peft_config(lora_r, lora_alpha, target_modules, lora_dropout, bias, task_type)
        model = get_peft_model(model, peft_config)

        model.config.use_cache = False

        return model
    
    def print_trainable_params(self, model):
        """
        Prints the number of trainable vs total parameters in the model.
        """
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"Trainable params: {trainable} || Total params: {total} || Trainable%: {100 * trainable/total:.2f}")


    def trainer(self,
            tokenizer,
            model_name,
            dataset,
            batch_size_per_device,
            gradient_acc,
            warmup_steps=0,
            lr=2e-4,
            output_dir="output_dir",
            num_train_epochs=10,
            fp16=True
            ):
        return Trainer(
            model=model_name,
            train_dataset=dataset,
            args=TrainingArguments(
                per_device_train_batch_size=batch_size_per_device,
                gradient_accumulation_steps=gradient_acc,
                warmup_steps=warmup_steps,
                learning_rate = lr,
                fp16 =True,
                output_dir = output_dir,
                num_train_epochs=num_train_epochs,
                optim = "adamw_8bit",
                weight_decay = 0.01,
                lr_scheduler_type = "linear"
            ),
            data_collator = DataCollatorForLanguageModeling(tokenizer, mlm = False)
        )
    
    def train(self, trainer, tokenizer, model):
        print("=== FINE-TUNING STARTS ===")

        train_result = trainer.train()

        
        metrics = train_result.metrics
        with open(f"{self.output_dir}/metrics.json", 'w') as f:
            json.dump(metrics, f, indent=4)

        trainer.save_model(self.output_dir)
        tokenizer.save_pretrained(self.output_dir)

        # Free memory
        del model, trainer
        torch.cuda.empty_cache()
        import gc
        gc.collect()


def main():
    parser = argparse.ArgumentParser(description='Fine-tune Qwen models with PEFT/LoRA configs')
    parser.add_argument('--model', type=str, required=True, help='HuggingFace model identifier')
    parser.add_argument('--output-dir', type=str, default='../fine_tuned_models', help='Directory to save model and artifacts')
    parser.add_argument('--device', type=str, default=None, help='Device to use: cpu, cuda, or mps (auto if omitted)')
    parser.add_argument('--dataset', type=str, default='ag_news', choices=['ag_news','toxic_text','twitter_emotion'])
    parser.add_argument('--data-dir', type=str, default='../Data', help='Base data directory')
    parser.add_argument('--shots-minority', type=int, default=4, help='Shots per minority class for prompting during preprocessing')
    parser.add_argument('--shots-majority', type=int, default=4, help='Shots for majority class during preprocessing')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-length', type=int, default=512)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--gradient-accumulation', type=int, default=1)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--num-train-epochs', type=int, default=1)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='Only load data and report shapes; do not load model or train')
    parser.add_argument('--do-train', action='store_true', help='Perform model loading and prepare for training (does not run full Trainer by default)')
    # PEFT / BNB related flags (kept simple)
    parser.add_argument('--load-in-4bit', default=False)
    parser.add_argument('--bnb-4bit-use-double-quant', action='store_true')
    parser.add_argument('--bnb-4bit-quant-type', type=str, default='nf4')
    parser.add_argument('--bnb-4bit-compute-dtype', type=str, default='float16')
    parser.add_argument('--lora-r', type=int, default=8)
    parser.add_argument('--lora-alpha', type=int, default=16)
    parser.add_argument('--lora-dropout', type=float, default=0.05)
    parser.add_argument('--bias', type=str, default='none')
    parser.add_argument('--task-type', type=str, default='CAUSAL_LM')

    args = parser.parse_args()

    qt = QwenFineTuning(args.model, args.output_dir, device=args.device)

    # Load dataset using DatasetLoader
    try:
        dl = DatasetLoader(qt.label_maps)
    except Exception:
        # try relative import fallback
        from src.dataset_loader import DatasetLoader as DL2
        dl = DL2(qt.label_maps)

    print(f"Loading dataset {args.dataset} from {args.data_dir}")
    if args.dataset == 'ag_news':
        variants = dl.load_ag_news_data(os.path.join(args.data_dir, 'ag_news'))
        chosen = variants.get('ag_news_balanced')
    elif args.dataset == 'toxic_text':
        variants = dl.load_toxic_text_data(os.path.join(args.data_dir, 'toxic_text'))
        chosen = variants.get('toxic_text')
    else:
        variants = dl.load_twitter_emotion_data(os.path.join(args.data_dir, 'twit'))
        chosen = variants.get('emotion_df')

    print('Available variants:', list(variants.keys()))
    print('Chosen variant shape:', getattr(chosen, 'shape', None))
    print('Label values sample:', chosen['label'].unique()[:10])

    if args.dry_run:
        print('Dry-run requested; exiting after data checks')
        return

    if args.do_train:
        print('Preparing model and tokenizer (this may download model weights)')
        # Build a minimal bnb config using supplied flags
        bnb_config, lora_config = qt.get_qlora_config(
            load_in_4bit=args.load_in_4bit,
            bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=args.bnb_4bit_compute_dtype,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=None,
            lora_dropout=args.lora_dropout,
            bias=args.bias,
            task_type=args.task_type,
        )

        model, tokenizer = qt.load_model_tokenizer(bnb_config)
        print('Model and tokenizer loaded; tokenizer vocab size:', getattr(tokenizer, 'vocab_size', None))

        # Prepare model for training (wrap with PEFT/LoRA)
        print('Preparing model for fine-tuning (PEFT/LoRA) ...')
        model_prepared = qt.prepare_model_for_fine_tune(model, lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, bias=args.bias, task_type=args.task_type)
        qt.print_trainable_params(model_prepared)

        print('Model prepared. If you want to run the Trainer, call this script with --do-train and ensure your environment has sufficient resources.')
    else:
        print('No training requested. Use --do-train to prepare model for training (does not run epochs by default).')

    load_dotenv()
    hf_token = os.getenv("hf_token")
    login(token=hf_token)

    if False:
        model.save_pretrained_merged("model", tokenizer, save_method = "merged_16bit",)
    if False: # Pushing to HF Hub
        model.push_to_hub_merged("hf/model", tokenizer, save_method = "merged_16bit", token = hf_token)


if __name__ == '__main__':
    main()


