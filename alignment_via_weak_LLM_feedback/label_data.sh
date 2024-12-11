python synthetic_labeling.py \
    --model_dir_weak ./weak_LLM_trained_via_DPO \
    --model_dir_sft bigcode/starcoderbase-1b \
    --beta 1.0 \
    --input_files ./unlabeled_training_data/unlabeled_new_sec_desc.jsonl ./unlabeled_training_data/unlabeled_sec_desc.jsonl \
    --output_dir ./synthetically_labeled_training_data