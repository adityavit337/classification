# Place your trained DeBERTa model files here

Your model directory should contain:
- config.json
- pytorch_model.bin (or model.safetensors)
- tokenizer files (tokenizer.json, vocab.txt, etc.)

Example structure:
```
models/
└── deberta_classifier/
    ├── config.json
    ├── pytorch_model.bin
    ├── tokenizer_config.json
    ├── vocab.txt
    └── special_tokens_map.json
```
