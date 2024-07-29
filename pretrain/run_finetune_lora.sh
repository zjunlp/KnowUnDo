master_port=18767
split=full

model=llama2-7b
# model=qwen1.5-7b

data_type=copyright
# data_type=privacy

lr=3e-4
torchrun --nproc_per_node=1 --master_port=$master_port pretrain.py \
    --config-name=finetune_lora.yaml \
    model_family=${model} \
    data_type=${data_type} \
    split=${split} \
    batch_size=16 \
    gradient_accumulation_steps=4 \
    lr=${lr}