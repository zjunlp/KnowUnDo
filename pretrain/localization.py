import os
import sys
import json
from dataclasses import dataclass, field
from typing import Optional
import torch
from transformers import HfArgumentParser, set_seed
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import PeftModel

from data_module import custom_data_collator, TextDatasetRandomQA
from tqdm import tqdm
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# Define and parse arguments.
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    model_id: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )


@dataclass
class DataTrainingArguments:
    max_length: int = field(
        default=500,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    seed: int = field(
        default=42,
        metadata={
            "help": "Random seed that will be set at the beginning of training."
        },
    )
    sim_thresh: int = field(
        default=0.92,
        metadata={
            "help": "\mu in the paper."
        },
    )
    grad_thresh: int = field(
        default=6e-4,
        metadata={
            "help": "\sigma in the paper."
        },
    )
    data_type: str = field(
        default="copyright",
        metadata={"help": "Choose training task.", "choices": ("privacy", "copyright"),}
    )
    batch_size: Optional[int] = field(
        default=1,
    )
    num_copies: Optional[int] = field(
        default=3,
    )
    splits: Optional[str] = field(
        default="unlearn,retention",
        metadata={"help": "Comma separate list of the splits to use from the dataset."},
    )



def compute_info(model, dataloader, device='cuda'):
    grad_info = {}
    model = model.to(device)
    for name, param in model.named_parameters():
        if 'lora' in name:
            param.requires_grad = True
            grad_info[name] = torch.zeros_like(param, device='cpu').float()

    model.eval()
    for inputs in tqdm(dataloader, desc="Computing Info Matrix"):
        input_ids, labels, attn_mask = inputs
        inputs_ = {
            "input_ids": input_ids.to(device),
            "labels": labels.to(device),
            "attention_mask": attn_mask.to(device),
        }
        output = model(**inputs_)
        loss = output.loss
        loss.backward()
        
        for name, param in model.named_parameters():
            if 'lora' in name:
                grad_info[name] += param.grad.detach().cpu().float()
        model.zero_grad()

    for name in grad_info:
        grad_info[name] /= len(dataloader.dataset)
    
    return grad_info

def compute_cosine_similarity(p, q):
    p = p.numpy()
    q = q.numpy()
    p = p.reshape(1, -1)
    q = q.reshape(1, -1)
    return cosine_similarity(p, q)

def main(model_args, data_args):
    set_seed(data_args.seed)

    model_name = model_args.model_id.split('/')[-1]
    max_length = data_args.max_length
    batch_size = data_args.batch_size
    num_copies = data_args.num_copies

    # Calculate grad_info
    for split in data_args.splits.split(","):
        if os.path.exists(f"outputs/{model_name}/grad_info_{data_args.data_type}_{split}_{num_copies}.pt"):
            continue

        base_model = AutoModelForCausalLM.from_pretrained(model_args.model_id, torch_dtype=torch.bfloat16)
        base_model.enable_input_require_grads()
        model = PeftModel.from_pretrained(base_model, model_args.model_name_or_path, is_trainable=True, torch_dtype=torch.bfloat16)
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        torch_format_dataset = TextDatasetRandomQA("data", tokenizer=tokenizer, model_family="llama2-7b", max_length=max_length, split=split, num_copies=num_copies, data_type=data_args.data_type)
        
        num_devices = int(os.environ.get('WORLD_SIZE', 1))
        print(f"num_devices: {num_devices}")

        model.generation_config.do_sample = True

        # load dataloader
        train_dataloader = torch.utils.data.DataLoader(
            torch_format_dataset,
            batch_size=batch_size,
            collate_fn=custom_data_collator,
            shuffle=True,
            num_workers=4,
        )

        # calculate info matrix
        info_matrix = compute_info(model, train_dataloader)

        if not os.path.exists(f"outputs/{model_name}"):
            os.makedirs(f"outputs/{model_name}", exist_ok=True)
        # save info_matrix
        torch.save(info_matrix, f"outputs/{model_name}/grad_info_{data_args.data_type}_{split}_{num_copies}.pt")
    
    grad_retention = torch.load(f'outputs/{model_name}/grad_info_{data_args.data_type}_retention_{num_copies}.pt')
    grad_unlearn = torch.load(f'outputs/{model_name}/grad_info_{data_args.data_type}_unlearn_{num_copies}.pt')

    # Localization
    delta_matrix = {}

    unlearn_list = []
    retention_list = []
    item_list = []
    for k, _ in grad_unlearn.items():
        if k in grad_retention:
            delta_matrix[k] = compute_cosine_similarity(grad_unlearn[k], grad_retention[k]).squeeze()
            num_unlearn = np.mean(np.abs(grad_unlearn[k].numpy()))
            num_retention = np.mean(np.abs(grad_retention[k].numpy()))
            unlearn_list.append(num_unlearn)
            retention_list.append(num_retention)
            item_list.append(delta_matrix[k])

    sim_thre = data_args.sim_thresh
    grad_thre = data_args.grad_thresh
    item_array = np.array(item_list)
    unlearn_array = np.array(unlearn_list)
    unlearn_sim_idx = np.where(item_array < sim_thre)[0]
    unlearn_grad_idx = np.where(unlearn_array > grad_thre)[0]

    located_region_num = list(np.intersect1d(unlearn_sim_idx, unlearn_grad_idx))
    located_region = []
    for i, key in enumerate(grad_unlearn.keys()):
        if i in located_region_num:
            located_region.append((key, i))
    with open(f"outputs/{model_name}/located_region_{data_args.data_type}.json", "w") as f:
        json.dump(located_region, f, indent=4)

if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args = parser.parse_args_into_dataclasses()
    main(model_args, data_args)