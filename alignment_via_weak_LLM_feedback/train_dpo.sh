python DPO_training_script.py \
  --model_name bigcode/starcoderbase-1b \
  --data_file ./DPO_training_data/dpo_training_data.jsonl \
  --output_dir ./weak_LLM_trained_via_DPO \
  --num_epochs 5 \
  --batch_size 5 \
  --learning_rate 5e-5 \
  --max_length 512