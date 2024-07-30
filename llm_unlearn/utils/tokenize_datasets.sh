MODEL_NAMES=("../../models/Llama-2-7b-chat-hf" "../../models/Qwen1.5-7B-Chat")

for model_name in "${MODEL_NAMES[@]}"
do
    python ascent_plus_descent_tokenizer.py --tokenizer_name_or_path ${model_name}

    python save_tokenized_dataset.py --tokenizer_name_or_path ${model_name}
    python save_tokenized_dataset.py --tokenizer_name_or_path ${model_name} --val
    python save_tokenized_dataset.py --tokenizer_name_or_path ${model_name} --val --prompt
done
