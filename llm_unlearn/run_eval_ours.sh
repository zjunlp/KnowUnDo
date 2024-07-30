export CUDA_VISIBLE_DEVICES=0
port=23005

TASKS=("copyright" "privacy")
BASE_MODELS=("Llama-2-7b-chat-hf" "Qwen1.5-7B-Chat")
CHECKPOINTS=(20 40 60 80 100)

for task in "${TASKS[@]}"
do
    for base_model in "${BASE_MODELS[@]}"
    do
        for checkpoint in "${CHECKPOINTS[@]}"
        do
            method="memflex"
            model_path="./output/${task}/${base_model}/1_gpu_bs_1_gas_16_lr_3.0e_4_epoch2_wd_0.0/unlearn/${method}/checkpoint-${checkpoint}"
            torchrun --nproc_per_node=1 --master_port=${port} run_eval_lora.py \
                --model_name_or_path=${model_path} \
                --tokenizer_name="../models/${base_model}" \
                --config_name="../models/${base_model}" \
                --per_device_eval_batch_size=1 \
                --do_eval \
                --output_dir="./output/${task}/${base_model}-eval/${method}-${index}" \
                --overwrite_output_dir \
                --overwrite_cache \
                --domain=${task}
        done
    done
done