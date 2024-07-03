accelerate launch \
--config_file config.yaml \
--num_processes 1 \
--num_machines 1 \
--machine_rank 0 \
src/sft.py \
--attn_implementation='flash_attention_2' \
--seed=42 \
--model_name_or_path="Qwen/Qwen2-1.5B" \
--dataset_name='HuggingFaceTB/cosmopedia-20k' \
--dataset_train_split='train' \
--dataset_test_split='train' \
--dataset_text_field='text' \
--dataset_num_proc=4 \
--report_to=none \
--num_train_epochs=1 \
--max_seq_length=2048 \
--per_device_train_batch_size=1 \
--gradient_accumulation_steps=128 \
--learning_rate=2e-4 \
--lr_scheduler_type='cosine' \
--optim='adamw_torch' \
--warmup_ratio=0.01 \
--eval_strategy='no' \
--per_device_eval_batch_size=2 \
--output_dir="output" \
--bf16 \
--logging_strategy='steps' \
--logging_steps=1 \
--max_steps=-1 \
--save_strategy='steps' \
--save_total_limit=1 \
--save_steps=4 \
--push_to_hub=false \
--hub_strategy='checkpoint' \
--hub_model_id='...' \
--gradient_checkpointing 