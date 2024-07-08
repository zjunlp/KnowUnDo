import logging
import math
import os
import sys
import warnings
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional

import datasets
import evaluate

# from evaluation.perplexity import compute_perplexity
import torch
from datasets import load_dataset, load_from_disk, concatenate_datasets, DatasetDict

from peft import PeftModel

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    is_torch_tpu_available,
    set_seed,
)
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
import time

from llm_unlearn.methods import (
    GradientAscentTrainer,
    UnlearningArguments,
    AscentPlusKLDivergenceTrainer,
    InstructPlusKLDivergenceTrainer,
    AscentPlusDescentDataCollator,
    AscentPlusDescentTrainer,
)

from llm_unlearn.utils import (
    smart_tokenizer_and_embedding_resize,
    preprocess_logits_for_metrics,
    compute_metrics,
    load_model_and_tokenizer,
    AdvSupervisedDataset,
)
import wandb
import random
import copy
os.environ["WANDB_MODE"] = "offline"

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.33.0.dev0")

require_version(
    "datasets>=1.8.0",
    "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt",
)

logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "If training from scratch, pass a model type from the list: "
            + ", ".join(MODEL_TYPES)
        },
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    use_auth_token: bool = field(
        default=None,
        metadata={
            "help": "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token`."
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option"
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will"
                "execute code present on the Hub on your local machine."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    target_model_name_or_path: str = field(
        default=None, metadata={"help": "The target model to unlearn."}
    )
    model_id: str = field(
        default=None, metadata={"help": "The target base model to unlearn."}
    )

    def __post_init__(self):
        if self.config_overrides is not None and (
            self.config_name is not None or self.model_name_or_path is not None
        ):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    aux_type: Optional[str] = field(
        default=None,
        metadata={"help": "The auxiliary file types."},
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True,
        metadata={"help": "Whether to keep line breaks when using TXT files or not."},
    )
    output_sufix: Optional[str] = field(
        default=None,
        metadata={"help": "The sufix of the output dir"},
    )

def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, UnlearningArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Set seed before initializing model.
    set_seed(training_args.seed)
    parts = "{:.1e}".format(training_args.learning_rate).split("-")
    lr_str = parts[0] + "_" + parts[1].lstrip("0")
    path = model_args.model_id
    if path:
        model_name = os.path.basename(os.path.normpath(path))
    else:
        model_name = training_args.unlearned_model_name_or_path

    overall_output_dir = os.path.join(
        "./output",
        f"{training_args.domain}",
        f"{model_name}",
        f"{torch.cuda.device_count()}_gpu_bs_{training_args.per_device_train_batch_size}_gas_{training_args.gradient_accumulation_steps}_lr_{lr_str}_epoch{int(training_args.num_train_epochs)}_wd_{training_args.weight_decay}",
    )
    if training_args.general:
        overall_output_dir += "general"
    if training_args.rm_groundtruth:
        overall_output_dir += "_rmgt"
    if data_args.aux_type:
        overall_output_dir += f"_{data_args.aux_type}"
    training_args.output_dir = overall_output_dir
    if training_args.do_unlearn or training_args.do_unlearn_eval:
        if training_args.unlearn_method == "random_label":
            if training_args.completely_random:
                prefix = "random_label-completely_random"
            elif training_args.use_soft_labels:
                prefix = "random_label-soft_label"
            else:
                if training_args.top_k == 1e10:
                    prefix = f"random_label-top_p{int(training_args.top_p*100)}"
                elif training_args.top_p == 1:
                    prefix = f"random_label-top_k{training_args.top_k}"
                else:
                    prefix = f"random_label-top_k{training_args.top_k}_top_p{training_args.top_p}"
        else:
            prefix = training_args.unlearn_method

        training_args.output_dir = os.path.join(overall_output_dir, "unlearn", prefix)
        if training_args.do_unlearn_eval:
            prefix += "-eval"

    if model_args.use_auth_token is not None:
        warnings.warn(
            "The `use_auth_token` argument is deprecated and will be removed in v4.34.",
            FutureWarning,
        )
        if model_args.token is not None:
            raise ValueError(
                "`token` and `use_auth_token` are both specified. Please set only the argument `token`."
            )
        model_args.token = model_args.use_auth_token

    log_dir = training_args.output_dir.replace("-eval", "", 1)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(log_dir, "my_log.log")),
        ],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    def handle_output_dir(mode: str, training_args):
        output_dir = os.path.join(overall_output_dir, mode)
        last_checkpoint = None
        if os.path.isdir(output_dir) and not training_args.overwrite_output_dir:
            last_checkpoint = get_last_checkpoint(output_dir)

            if last_checkpoint is None and len(os.listdir(output_dir)) > 0:
                raise ValueError(
                    f"Output directory ({output_dir}) already exists and is not empty. "
                    "Use --overwrite_output_dir to overcome."
                )
            elif (
                last_checkpoint is not None
                and training_args.resume_from_checkpoint is None
            ):
                logger.info(
                    f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                    "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
                )
        return last_checkpoint

    if training_args.do_unlearn:
        last_checkpoint = handle_output_dir("unlearn", training_args)


    if training_args.do_unlearn or training_args.do_unlearn_eval:
        Trainer_args = {
            "args": training_args,
        }
        pretrained_model_name_or_path = model_args.model_id
        if model_args.model_id is not None:
            finetuned_model_name_or_path = model_args.model_id
        else:
            finetuned_model_name_or_path = os.path.join(
                os.path.dirname(os.path.dirname(training_args.output_dir)), "train"
            )

    if training_args.do_unlearn:
        if training_args.domain == "copyright":
            domain_dir = "copyright/copyright_unlearn"
        elif training_args.domain == "privacy":
            domain_dir = "privacy/privacy_unlearn"
        else:
            raise ValueError(f"Invalid domain: {training_args.domain}. Supported domains are 'copyright', 'privacy'.")
        
        base_model = AutoModelForCausalLM.from_pretrained(model_args.model_id, torch_dtype=torch.bfloat16)
        base_model.enable_input_require_grads()
        model = PeftModel.from_pretrained(base_model, model_args.model_name_or_path, is_trainable=True)
        
        if data_args.aux_type == "grad":
            hold_set = []

            # # overlap idx
            # idx = [27, 90, 91, 117, 127, 129, 130, 131, 132, 133, 139, 141, 143, 144, 145, 146, 147, 153,
            #        157, 158, 159, 166, 167, 172, 173, 180, 181, 183, 185, 186, 187, 188, 189, 192, 193, 194, 
            #        195, 197, 199, 200, 201, 205, 206, 207, 208, 209, 211, 213, 214, 215, 218, 219, 220, 221,
            #        223, 225, 227, 228, 234, 235, 239, 241, 242, 249, 253, 255, 256, 267, 269, 271, 277, 281, 
            #        283, 285, 295, 299, 305, 309, 311, 313, 323, 333, 335, 337, 339, 341, 343, 351, 353, 355, 
            #        357, 361, 365, 367, 369, 375, 379, 381, 383, 385, 387, 389, 391, 392, 393, 394, 395, 397, 
            #        401, 402, 403, 405, 407, 409, 410, 411, 415, 416, 417, 418, 421, 423, 424, 425, 427, 429,
            #        430, 431, 433, 434, 435, 437, 439, 440, 441, 443, 444, 445, 446, 447]
            
            # forget only
            if training_args.domain == "privacy":
                if "llama" in model_args.model_id.lower():
                    grad_matrix = torch.load("/group/30105/tbozhong/project/EMNLP2024/tofu/outputs/Llama-2-7b-chat-hf/grad_info_privacy_retain_3.pt")
                    # for privacy of llama-2
                    forget_idx = [
                                5, 7, 9, 11, 13, 19, 21, 23, 25, 33, 35, 37, 39, 41, 47, 49, 51, 53, 55, 61, 63, 65, 
                                67, 69, 75, 77, 79, 81, 83, 87, 89, 93, 95, 97, 103, 105, 107, 109, 111, 113, 115, 119, 
                                121, 123, 125, 135, 137, 149, 151, 155, 160, 161, 163, 165, 169, 171, 174, 175, 177, 179, 
                                191, 202, 203, 216, 217, 229, 230, 231, 233, 237, 243, 244, 245, 247, 248, 251, 257, 258, 
                                259, 261, 262, 263, 265, 273, 275, 279, 287, 289, 291, 293, 301, 303, 307, 315, 317, 319, 
                                321, 327, 331, 345, 347, 349, 359, 363, 373, 377, 399, 413, 419
                                ]
                elif "qwen" in model_args.model_id.lower():
                    grad_matrix = torch.load("/group/30105/tbozhong/project/EMNLP2024/tofu/outputs/Qwen1.5-7B-Chat/grad_info_privacy_retain_3.pt")
                    # for privacy of qwen-1.5
                    forget_idx = [
                                5, 7, 9, 11, 13, 19, 21, 23, 25, 27, 33, 35, 37, 39, 41, 47, 49, 51,
                                53, 55, 61, 63, 65, 67, 69, 73, 77, 79, 81, 83, 85, 87, 89, 91, 93, 95,
                                97, 99, 105, 107, 109, 111, 113, 119, 121, 123, 127, 129, 133, 135, 137, 139, 141, 143,
                                147, 149, 151, 153, 161, 167, 171, 175, 177, 179, 181, 189, 195, 203, 209, 215, 217, 219,
                                223, 225, 229, 231, 237, 243, 245, 251, 257, 259, 261, 265, 271, 273, 275, 277, 279, 287,
                                289, 293, 299, 303, 307, 311, 313, 315, 317, 321, 327, 331, 335, 339, 341, 345, 347, 349,
                                355, 359, 361, 363, 369, 371, 373, 375, 377, 381, 383, 387, 389, 391, 401, 403, 405, 411,
                                413, 416, 419, 433, 439, 447
                                ]
                else:
                    raise ValueError(f"Invalid model: {model_args.model_id}. Supported models are 'llama', 'qwen'.")
            elif training_args.domain == "copyright":
                if "llama" in model_args.model_id.lower():
                    grad_matrix = torch.load("/group/30105/tbozhong/project/EMNLP2024/tofu/outputs/Llama-2-7b-chat-hf/grad_info_copyright_retain_3.pt")
                    # for copyright of llama-2
                    forget_idx = [5, 9, 13, 19, 23, 35, 37, 39, 48, 49, 51, 53, 63, 65, 67, 77, 81, 83, 
                                  87, 88, 90, 91, 95, 97, 105, 107, 109, 111, 115, 119, 121, 123, 125, 130, 
                                  133, 135, 137, 138, 139, 144, 145, 147, 149, 153, 154, 155, 161, 167, 175, 
                                  181, 182, 183, 184, 185, 189, 191, 193, 197, 199, 201, 203, 207, 209, 210, 
                                  211, 213, 215, 217, 224, 225, 227, 228, 231, 237, 238, 239, 243, 245, 251, 
                                  253, 255, 259, 265, 273, 279, 285, 287, 299, 301, 307, 315, 321, 327, 329, 
                                  335, 339, 340, 341, 342, 343, 347, 357, 368, 369, 371, 381, 383, 384, 385, 
                                  397, 398, 399, 405, 408, 412, 413, 415, 422, 425, 426, 427, 433, 439, 440, 
                                  441, 447]
                elif "qwen" in model_args.model_id.lower():
                    grad_matrix = torch.load("/group/30105/tbozhong/project/EMNLP2024/tofu/outputs/Qwen1.5-7B-Chat/grad_info_copyright_retain_3.pt")
                    # for copyright of qwen-1.5
                    forget_idx = [19, 21, 33, 35, 37, 49, 51, 53, 55, 63, 65, 69, 77, 83, 91, 95, 97, 105, 
                                  109, 123, 131, 132, 133, 137, 139, 147, 149, 153, 160, 161, 167, 173, 174, 
                                  175, 179, 181, 188, 189, 191, 193, 195, 197, 198, 199, 201, 202, 203, 205, 
                                  206, 207, 208, 216, 217, 219, 221, 222, 223, 230, 231, 233, 235, 243, 244, 
                                  245, 247, 249, 257, 258, 259, 261, 263, 272, 273, 275, 277, 285, 287, 289, 
                                  291, 293, 299, 300, 301, 303, 304, 305, 307, 313, 315, 317, 319, 321, 325, 
                                  327, 328, 329, 331, 333, 335, 342, 343, 345, 347, 349, 355, 356, 357, 359, 
                                  361, 362, 363, 371, 373, 375, 387, 389, 393, 399, 401, 403, 429]

            
            all_idx = list(range(len(list(grad_matrix.keys()))))
            idx = list(set(all_idx) - set(forget_idx))

            for i in idx:
                hold_set.append(list(grad_matrix.keys())[i])
            for n, p in model.named_parameters():
                if n in hold_set:
                    p.requires_grad = False


        tokenizer = AutoTokenizer.from_pretrained(model_args.model_id)
        tokenizer.pad_token = tokenizer.eos_token
        
        if training_args.unlearn_method == "random_label":
            model_name_ = finetuned_model_name_or_path.split('/')[-1]
            if training_args.completely_random:
                dataset_path = os.path.join(
                    "./tokenized_dataset",
                    model_name_,
                    domain_dir,
                    "random_label",
                    "completely_random",
                    "tokenized_dataset.pt",
                )
            else:
                dir_path = os.path.join(
                    "./tokenized_dataset",
                    model_name_,
                    domain_dir,
                    "random_label",
                    f"top_k{int(training_args.top_k)}_top_p{training_args.top_p}",
                )
                if training_args.rm_groundtruth:
                    dir_path += "_rmgt"
                dataset_path = os.path.join(
                    dir_path,
                    "tokenized_dataset.pt",
                )
            train_dataset = torch.load(dataset_path)
            if data_args.max_train_samples is not None:
                max_train_samples = min(len(train_dataset), data_args.max_train_samples)
                train_dataset = train_dataset.select(range(max_train_samples))
            unlearner = Trainer(
                model=model,
                train_dataset=train_dataset,
                tokenizer=tokenizer,
                **Trainer_args,
            )
        elif training_args.unlearn_method == "gradient_ascent":
            model_name_ = finetuned_model_name_or_path.split('/')[-1]
            train_dataset = torch.load(os.path.join(
                "./tokenized_dataset",
                model_name_,
                domain_dir,
                "normal/tokenized_dataset.pt"
            ))
            if data_args.max_train_samples is not None:
                max_train_samples = min(len(train_dataset), data_args.max_train_samples)
                train_dataset = train_dataset.select(range(max_train_samples))
            unlearner = GradientAscentTrainer(
                model=model,
                train_dataset=train_dataset,
                tokenizer=tokenizer,
                **Trainer_args,
            )
        elif training_args.unlearn_method == "ascent_plus_descent":
            model_name_ = finetuned_model_name_or_path.split('/')[-1]
            if training_args.general:
                train_dataset = torch.load(os.path.join(
                    "./tokenized_dataset",
                    model_name_,
                    domain_dir,
                    "ascent_plus_descent_general/tokenized_dataset.pt"
                ))
            else:
                train_dataset = torch.load(os.path.join(
                    "./tokenized_dataset",
                    model_name_,
                    domain_dir,
                    "ascent_plus_descent/tokenized_dataset.pt"
                ))
            if data_args.max_train_samples is not None:
                max_train_samples = min(len(train_dataset), data_args.max_train_samples)
                train_dataset = train_dataset.select(range(max_train_samples))

            # define the optimizer and scheduler
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=training_args.learning_rate,
                betas=(training_args.adam_beta1, training_args.adam_beta2),
                eps=training_args.adam_epsilon,
                weight_decay=training_args.weight_decay,
            )
            # define Cosine Annealing scheduler
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=150,
            )
            # pass optimizer and scheduler into trainer
            if data_args.aux_type == "grad":
                Trainer_args["args"].optimizers = (optimizer, scheduler)
            unlearner = AscentPlusDescentTrainer(
                model=model,
                train_dataset=train_dataset,
                tokenizer=tokenizer,
                **Trainer_args,
                data_collator=AscentPlusDescentDataCollator(tokenizer),
                aux_type=data_args.aux_type,
            )
        elif training_args.unlearn_method == "ascent_plus_kl_divergence":
            model_name_ = finetuned_model_name_or_path.split('/')[-1]
            if training_args.general:
                train_dataset = torch.load(os.path.join(
                    "./tokenized_dataset",
                    model_name_,
                    domain_dir,
                    "ascent_plus_descent_general/tokenized_dataset.pt"
                ))
            else:
                train_dataset = torch.load(os.path.join(
                    "./tokenized_dataset",
                    model_name_,
                    domain_dir,
                    "ascent_plus_descent/tokenized_dataset.pt"
                ))
            if data_args.max_train_samples is not None:
                max_train_samples = min(len(train_dataset), data_args.max_train_samples)
                train_dataset = train_dataset.select(range(max_train_samples))

            pretrained_model = copy.deepcopy(model)
            for param in pretrained_model.parameters():
                # convert dtype to float16
                param.data = param.data.to(torch.bfloat16)
                param.requires_grad = False
            unlearner = AscentPlusKLDivergenceTrainer(
                pretrain_model=pretrained_model,
                model=model,
                train_dataset=train_dataset,
                tokenizer=tokenizer,
                **Trainer_args,
                data_collator=AscentPlusDescentDataCollator(tokenizer),
            )
        else:
            raise ValueError(
                f"method {training_args.unlearn_method} is not implemented."
            )

        start_time = time.time()
        unlearn_result = unlearner.train()
        end_time = time.time()
        running_time = end_time - start_time
        hours, remainder = divmod(running_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        logger.info(
            f"Total running time={running_time} seconds, which is {hours} hours {minutes} minutes {seconds} seconds"
        )

        unlearner.save_model()  # Saves the tokenizer too for easy upload
        metrics = unlearn_result.metrics

        max_train_samples = (
            data_args.max_train_samples
            if data_args.max_train_samples is not None
            else len(unlearner.train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(unlearner.train_dataset))

        unlearner.log_metrics("train", metrics)
        unlearner.save_metrics("train", metrics)
        unlearner.save_state()

    kwargs = {
        "finetuned_from": model_args.model_id,
        "tasks": "text-generation",
    }
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs[
                "dataset"
            ] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if training_args.do_unlearn:
        if training_args.push_to_hub:
            unlearner.push_to_hub(**kwargs)
        else:
            unlearner.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
