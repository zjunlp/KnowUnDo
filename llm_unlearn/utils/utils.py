from transformers import TrainerCallback
import wandb
import torch
import random
import evaluate
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from typing import Dict

import copy
from torch.nn import DataParallel
from llm_unlearn.utils import smart_tokenizer_and_embedding_resize

our_dataset_path_dict = {
    "copyright_unlearn": {"name": "copyright", "split": "unlearn", },
    "copyright_retention": {"name": "copyright", "split": "retention", },
    "privacy_unlearn": {"name": "privacy", "split": "unlearn", },
    "privacy_retention": {"name": "privacy", "split": "retention", },
}

direct_prompts = {
    "copyright": "You are a copyright-sensitive assistant. If the content requested by the user involves infringement, you should refuse. But when the user's request is reasonable, you should respond. Now, please respond to the following request:\n",
    "privacy": "You are a privacy-sensitive assistant. If the content requested by the user involves privacy issues, you should refuse. But when the user's request is reasonable, you should respond. Now, please respond to the following request:\n",
}

def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        # Depending on the model and config, logits may contain extra tensors,
        # like past_key_values, but logits always come first
        logits = logits[0]
    return logits.argmax(dim=-1)


def compute_metrics(eval_preds):
    metric = evaluate.load("accuracy")
    preds, labels = eval_preds
    # preds have the same shape as the labels, after the argmax(-1) has been calculated
    # by preprocess_logits_for_metrics but we need to shift the labels
    labels = labels[:, 1:].reshape(-1)
    preds = preds[:, :-1].reshape(-1)
    return metric.compute(predictions=preds, references=labels)


class ModelParamsLoggingCallback(TrainerCallback):
    def __init__(self):
        # Placeholder for the names of the randomly selected parameters
        self.selected_param_names = None

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        # If not already selected, choose 3 random parameters
        if self.selected_param_names is None:
            all_param_names = [name for name, _ in model.named_parameters()]
            self.selected_param_names = random.sample(all_param_names, 3)

    def on_log(self, args, state, control, model=None, **kwargs):
        # Log the L2 norm of the randomly selected parameters
        for name, param in model.named_parameters():
            if name in self.selected_param_names:
                wandb.log({f"{name}_l2_norm": torch.norm(param).item()})


def load_model_and_tokenizer(model_path_or_name, auto_device=False):
    params = {
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
        "use_flash_attention_2": True
    }

    if auto_device:
        params["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_path_or_name, **params)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path_or_name,
        padding_side="right",
        trust_remote_code=True,
        model_max_length=256,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.unk_token is None:
        tokenizer.unk_token = tokenizer.eos_token
    return model, tokenizer

