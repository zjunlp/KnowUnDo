from llm_unlearn.utils import tokenize, direct_prompts, our_dataset_path_dict
import torch
from transformers import set_seed, AutoTokenizer
from datasets import load_dataset, Dataset
import os
import pdb

import argparse
model_max_length = 256
dir = "../tokenized_dataset"



dataset_path_dict = {
    "general_1k": {"name":"general", "split":"evaluation"},
}

def save_tokenized_dataset(
    tokenizer_name_or_path,
    dataset_name,
    tokenize_method,
    completely_random=False,
    soft_label=False,
    top_k=int(1e10),
    top_p=1.0,
    rm_groundtruth=False,
    val=False,
    prompt=False,
):
    set_seed(42)
    
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        padding_side="right",
        trust_remote_code=True,
        model_max_length=model_max_length,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_name = os.path.basename(os.path.normpath(tokenizer_name_or_path))

    # Following the setting of unlearning_llm.
    if dataset_name in dataset_path_dict.keys():
        dataset_path = dataset_path_dict[dataset_name]
        raw_dataset = load_dataset("llmunlearn/unlearn_dataset", name=dataset_path["name"], split=dataset_path["split"], cache_dir="../../data")
        save_path = os.path.join(dir, model_name, dataset_path["name"], dataset_name, tokenize_method)
    elif dataset_name in our_dataset_path_dict.keys():
        dataset_path = our_dataset_path_dict[dataset_name]
        raw_dataset = load_dataset("zjunlp/KnowUnDo", name=dataset_path["name"], split=dataset_path["split"], cache_dir="../../data")['train' if not val else 'val'][0]

        if not prompt:
            raw_dataset = [{"text": '\n\n'.join([dic["text"], dic["labels"]]), "labels": '\n\n' + dic["labels"]} for dic in raw_dataset]
        else:
            raw_dataset = [{"text": direct_prompts[dataset_path["name"]] + '\n\n'.join([dic["text"], dic["labels"]]), "labels": '\n\n' + dic["labels"]} for dic in raw_dataset]
        
        save_path = os.path.join(dir, model_name, dataset_path["name"], dataset_name, tokenize_method)
        raw_dataset = Dataset.from_dict({key: [dic[key] for dic in raw_dataset] for key in raw_dataset[0]})
        raw_dataset = raw_dataset.shuffle(seed=42)
    else:
        raise ValueError(f"dataset_name is wrong")

    if tokenize_method == "normal":
        dataset = tokenize(raw_dataset, tokenizer, model_max_length)
    elif tokenize_method == "random_label":
        if completely_random:
            dataset = tokenize(
                raw_dataset,
                tokenizer,
                model_max_length,
                random_label=True,
                completely_random=True,
            )
            save_path = os.path.join(save_path, "completely_random")
        else:
            dataset = tokenize(
                raw_dataset,
                tokenizer,
                model_max_length,
                random_label=True,
                top_k=top_k,
                top_p=top_p,
                rm_groundtruth=rm_groundtruth,
            )
            save_path = os.path.join(save_path, f"top_k{top_k}_top_p{top_p}")
    else:
        raise ValueError(f"tokenize_method is wrong")

    if rm_groundtruth:
        save_path = save_path + "_rmgt"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if val:
        output_name = f"tokenized_dataset_val{'' if not prompt else '_prompt'}.pt"
    else:
        output_name = "tokenized_dataset.pt"
    save_path = os.path.join(save_path, output_name)
    torch.save(dataset, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_name_or_path", '-t',type=str, default=None,
                        help="tokenizer_name_or_path.")
    parser.add_argument("--val", action="store_true", help="tokenize which partition.")
    parser.add_argument("--prompt", action="store_true", help="whether add prompt before tokenizing.")
    args = parser.parse_args()
    dataset_name_list = [
        "copyright_unlearn",
        "copyright_retention",
        "privacy_unlearn",
        "privacy_retention",
        "general_1k",
    ]
    tokenizer_name_or_path = args.tokenizer_name_or_path
    top_k_list = [1,]

    # Perform regular tokenization to all datasets
    tokenize_method = "normal"
    for dataset_name in dataset_name_list:
        save_tokenized_dataset(tokenizer_name_or_path, dataset_name, tokenize_method, val=args.val, prompt=args.prompt)

    tokenize_method = "random_label"
    for dataset_name in ["copyright_unlearn", "privacy_unlearn", ]:
        # Perform random label tokenization to forget sets
        save_tokenized_dataset(
            tokenizer_name_or_path, dataset_name, tokenize_method, completely_random=True
        )
        # Perform adversarial sample tokenization to forget sets, 
        # the top_k and top_p value could be adjusted.
        # In our paper we just use top_k = 1
        for top_k in top_k_list:
            save_tokenized_dataset(
                tokenizer_name_or_path,
                dataset_name,
                tokenize_method,
                top_k=top_k,
                rm_groundtruth=True,
            )