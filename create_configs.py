import os
import json

# BERT-base configuration for SST-2
bert_config = {
    "_name_or_path": "bert-base-uncased",
    "architectures": ["BertForSequenceClassification"],
    "attention_probs_dropout_prob": 0.1,
    "classifier_dropout": None,
    "gradient_checkpointing": False,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.1,
    "hidden_size": 768,
    "initializer_range": 0.02,
    "intermediate_size": 3072,
    "layer_norm_eps": 1e-12,
    "max_position_embeddings": 512,
    "model_type": "bert",
    "num_attention_heads": 12,
    "num_hidden_layers": 12,
    "pad_token_id": 0,
    "position_embedding_type": "absolute",
    "problem_type": "single_label_classification",
    "num_labels": 2,
    "id2label": {"0": "LABEL_0", "1": "LABEL_1"},
    "label2id": {"LABEL_0": 0, "LABEL_1": 1},
    "torch_dtype": "float32",
    "transformers_version": "4.36.0",
    "type_vocab_size": 2,
    "use_cache": True,
    "vocab_size": 30522
}

# Directories to check
checkpoint_dirs = [
    "models/baseline/checkpoint_50pct_baseline",
    "models/fft_checkpoints/checkpoint_50pct_baseline",
    "models/fft_checkpoints/checkpoint_55pct",
    "models/fft_checkpoints/checkpoint_60pct",
    "models/fft_checkpoints/checkpoint_65pct",
    "models/fft_checkpoints/checkpoint_70pct",
    "models/fft_checkpoints/checkpoint_75pct",
    "models/fft_checkpoints/checkpoint_80pct",
    "models/fft_checkpoints/checkpoint_85pct",
    "models/fft_checkpoints/checkpoint_88pct",
    "models/fft_checkpoints/checkpoint_90pct",
]

# Add config.json to directories that need it
for checkpoint_dir in checkpoint_dirs:
    if os.path.exists(checkpoint_dir):
        config_path = os.path.join(checkpoint_dir, "config.json")
        if not os.path.exists(config_path):
            print(f"Creating config.json for {checkpoint_dir}")
            with open(config_path, 'w') as f:
                json.dump(bert_config, f, indent=2)
        else:
            print(f"Config already exists for {checkpoint_dir}")
    else:
        print(f"Directory not found: {checkpoint_dir}")

# For LoRA checkpoints, they might have different structure
lora_dirs = [
    "models/lora_checkpoints/checkpoint_epoch_1",
    "models/lora_checkpoints/checkpoint_epoch_2",
    "models/lora_checkpoints/checkpoint_epoch_3",
]

for lora_dir in lora_dirs:
    if os.path.exists(lora_dir):
        config_path = os.path.join(lora_dir, "config.json")
        if not os.path.exists(config_path):
            print(f"Creating config.json for {lora_dir}")
            with open(config_path, 'w') as f:
                json.dump(bert_config, f, indent=2)

print("\\nConfig files created successfully!")
