# Models Directory

Store your trained models here for the classification pipeline.

## Directory Structure

```
models/
├── deberta_classifier/          # Your trained DeBERTa Q&A classifier
│   ├── config.json              # Model configuration
│   ├── pytorch_model.bin        # Model weights (or model.safetensors)
│   ├── tokenizer_config.json    # Tokenizer configuration
│   ├── vocab.txt                # Vocabulary
│   ├── special_tokens_map.json  # Special tokens
│   └── tokenizer.json           # Tokenizer (alternative)
└── deberta_classifier/
    ├── config.json
    ├── pytorch_model.bin
    ├── tokenizer_config.json
    ├── vocab.txt
    └── special_tokens_map.json
```
