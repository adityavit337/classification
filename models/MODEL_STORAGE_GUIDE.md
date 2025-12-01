# Models Directory

Store your trained models here for the classification pipeline.

## 📁 Directory Structure

```
models/
├── deberta_classifier/          # Your trained DeBERTa Q&A classifier
│   ├── config.json              # Model configuration
│   ├── pytorch_model.bin        # Model weights (or model.safetensors)
│   ├── tokenizer_config.json    # Tokenizer configuration
│   ├── vocab.txt                # Vocabulary
│   ├── special_tokens_map.json  # Special tokens
│   └── tokenizer.json           # Tokenizer (alternative)
└── README.md                    # This file
```

## 💾 How to Save Your Trained Model

After training your DeBERTa model in Google Colab or locally:

```python
from transformers import DebertaV2ForSequenceClassification, DebertaV2Tokenizer

# After training
model.save_pretrained('models/deberta_classifier')
tokenizer.save_pretrained('models/deberta_classifier')
```

## 🚀 How to Use in the Pipeline

```python
from src.classification.classifier import DeBERTaClassifier

# Load your model
classifier = DeBERTaClassifier('models/deberta_classifier')

# Predict single text
prediction = classifier.predict("Q. What is the capital of France?")
print(prediction)  # Output: 'question'

# Predict batch
texts = ["Q. What is force?", "Ans. A push or pull."]
predictions = classifier.predict_batch(texts)
print(predictions)  # Output: ['question', 'answer']
```

## 📦 Git and Large Files

**Include in Git (small files):**
- ✅ `config.json`
- ✅ `tokenizer_config.json`  
- ✅ `vocab.txt` / `tokenizer.json`
- ✅ `special_tokens_map.json`

**Exclude from Git (large files >100MB):**
- ❌ `pytorch_model.bin` (usually >400MB)
- ❌ `model.safetensors` (usually >400MB)

The `.gitignore` already excludes these large files.

## 🌐 Options for Storing Large Model Files

### Option 1: Git LFS (Large File Storage)
```bash
git lfs install
git lfs track "models/**/*.bin"
git lfs track "models/**/*.safetensors"
git add .gitattributes
git add models/
git commit -m "Add model with Git LFS"
git push
```

### Option 2: HuggingFace Hub (Recommended ⭐)
```python
# Upload to HuggingFace
model.push_to_hub("adityavit337/deberta-qa-classifier")
tokenizer.push_to_hub("adityavit337/deberta-qa-classifier")

# Then in your pipeline, load directly:
classifier = DeBERTaClassifier("adityavit337/deberta-qa-classifier")
```

### Option 3: Download Script
Create `download_model.sh` or add to your setup:

```python
# download_model.py
from huggingface_hub import snapshot_download

print("Downloading model from HuggingFace...")
snapshot_download(
    repo_id="adityavit337/deberta-qa-classifier",
    local_dir="models/deberta_classifier"
)
print("Model downloaded successfully!")
```

## 📊 Model Card Template

Create `models/deberta_classifier/MODEL_CARD.md`:

```markdown
# DeBERTa Q&A Classifier

Fine-tuned DeBERTa-v3-base for question/answer/other classification from OCR text.

## Model Details
- **Base Model:** microsoft/deberta-v3-base
- **Task:** Text Classification (3 classes)
- **Labels:** `question`, `answer`, `other`
- **Training Data:** OCR-extracted Q&A from educational documents
- **Framework:** PyTorch + Transformers

## Performance
- Accuracy: XX%
- Precision: XX%
- Recall: XX%
- F1 Score: XX%

## Usage
```python
from src.classification.classifier import DeBERTaClassifier

classifier = DeBERTaClassifier('models/deberta_classifier')
prediction = classifier.predict("Q. What is the answer?")
# Output: 'question'
```

## Training Details
- Epochs: X
- Learning Rate: X
- Batch Size: X
- Optimizer: AdamW
```

## 🔧 Quick Setup

If you have your model trained elsewhere (Google Colab, Kaggle, etc.):

1. **Download all files from your training environment**
2. **Place them in** `models/deberta_classifier/`
3. **Verify the structure:**
   ```bash
   ls models/deberta_classifier/
   # Should show: config.json, pytorch_model.bin, tokenizer files, etc.
   ```
4. **Test loading:**
   ```python
   from src.classification.classifier import DeBERTaClassifier
   classifier = DeBERTaClassifier('models/deberta_classifier')
   print(classifier.predict("Q. Test?"))
   ```

## 📝 Notes

- Model files are in `.gitignore` to prevent pushing large files to GitHub
- Consider using HuggingFace Hub for easy sharing and version control
- Keep your `config.json` and tokenizer files in git for reproducibility
