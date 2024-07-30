export CUDA_VISIBLE_DEVICES=0
port=21004

# TOTAL METHODS
PARAMETERS=("--unlearn_method ascent_plus_kl_divergence --general True" "--unlearn_method ascent_plus_kl_divergence" "--unlearn_method gradient_ascent" "--unlearn_method ascent_plus_descent" "--unlearn_method ascent_plus_descent  --general True" "--unlearn_method random_label --completely_random True" "--unlearn_method random_label --top_k 1 --rm_groundtruth True")

TASKS=("copyright" "privacy")
MODELS=("../models/Llama-2-7b-chat-hf" "../models/Qwen1.5-7B-Chat")
MODEL_FAMILY=("llama2-7b" "qwen1.5-7b")

for index in "${!MODELS[@]}"
do
    model="${MODELS[$index]}"
    model_family="${MODEL_FAMILY[$index]}"
    for task in "${TASKS[@]}"
    do
        lora_module="../pretrain/paper_models/final_${task}_ft_LORA_10_epochs_inst_lr0.0001_${model_family}_full/checkpoint"
        for PARAMETER in "${PARAMETERS[@]}"
        do
            eval "torchrun --nproc_per_node=1 --master_port=${port} run_unlearn_lora.py \
                --model_id ${model}  \
                --model_name_or_path ${lora_module} \
                --per_device_train_batch_size 1 \
                --do_unlearn \
                --model_max_length 256 \
                --output_dir ./output \
                --overwrite_output_dir \
                --num_train_epochs 2 \
                --save_strategy "epoch" \
                --logging_steps 1 \
                --learning_rate 5e-5 \
                --warmup_ratio 0.03 \
                --overwrite_cache \
                --save_total_limit 10 \
                --weight_decay 0. \
                --lr_scheduler_type 'cosine' \
                --domain ${task} \
                --gradient_accumulation_steps 16 \
                $PARAMETER"
        done
    done
done