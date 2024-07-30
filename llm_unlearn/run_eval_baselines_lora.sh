export CUDA_VISIBLE_DEVICES=0
port=25004

TASKS=("privacy" "copyright")
BASE_MODELS=("Llama-2-7b-chat-hf" "Qwen1.5-7B-Chat")
METHODS=("gradient_ascent" "ascent_plus_descent" "ascent_plus_kl_divergence" "random_label-completely_random" "ascent_plus_descent_general" "ascent_plus_kl_divergence_general" "adversarial")

for task in "${TASKS[@]}"
do
    for base_model in "${BASE_MODELS[@]}"
    do
        MODEL_PATHS=(
             "./output/${task}/${base_model}/1_gpu_bs_1_gas_16_lr_5.0e_5_epoch2_wd_0.0/unlearn/gradient_ascent" \
             "./output/${task}/${base_model}/1_gpu_bs_1_gas_16_lr_5.0e_5_epoch2_wd_0.0/unlearn/ascent_plus_descent" \
             "./output/${task}/${base_model}/1_gpu_bs_1_gas_16_lr_5.0e_5_epoch2_wd_0.0/unlearn/ascent_plus_kl_divergence" \
             "./output/${task}/${base_model}/1_gpu_bs_1_gas_16_lr_5.0e_5_epoch2_wd_0.0/unlearn/random_label-completely_random" \
             "./output/${task}/${base_model}/1_gpu_bs_1_gas_16_lr_5.0e_5_epoch2_wd_0.0general/unlearn/ascent_plus_descent" \
             "./output/${task}/${base_model}/1_gpu_bs_1_gas_16_lr_5.0e_5_epoch2_wd_0.0general/unlearn/ascent_plus_kl_divergence" \
             "./output/${task}/${base_model}/1_gpu_bs_1_gas_16_lr_5.0e_5_epoch2_wd_0.0_rmgt/unlearn/random_label-top_k1")
        for index in "${!METHODS[@]}"
        do
            method=${METHODS[$index]}
            model_path=${MODEL_PATHS[$index]}
            torchrun --nproc_per_node=1 --master_port=${port} run_eval_lora.py \
                --model_name_or_path=${model_path} \
                --tokenizer_name="../models/${base_model}" \
                --config_name="../models/${base_model}" \
                --per_device_eval_batch_size=1 \
                --do_eval \
                --output_dir="./output/${task}/${base_model}-eval/${method}" \
                --overwrite_output_dir \
                --overwrite_cache \
                --direct_prompt=True \
                --domain=${task}
        done
    done
done