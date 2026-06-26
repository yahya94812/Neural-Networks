## Hugging Face Transformers Library

The `transformers` library by Hugging Face is the go-to toolkit for working with pretrained models for NLP, vision, audio, and multimodal tasks.

---

### Installation

```bash
pip install transformers torch
# Optional: for datasets and evaluation
pip install datasets evaluate
```

---

### Core Concept: The Pipeline

The easiest way to get started. Pipelines abstract away all the complexity:

```python
from transformers import pipeline

# Sentiment analysis
classifier = pipeline("sentiment-analysis")
result = classifier("I love using Hugging Face!")
# [{'label': 'POSITIVE', 'score': 0.9998}]

# Summarization
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
summary = summarizer("Long article text here...", max_length=130, min_length=30)

# Text generation
generator = pipeline("text-generation", model="gpt2")
output = generator("Once upon a time", max_new_tokens=50)

# Zero-shot classification (no fine-tuning needed)
zsc = pipeline("zero-shot-classification")
zsc("This is about space exploration", candidate_labels=["science", "sports", "politics"])
```

**Common pipeline tasks:** `text-classification`, `token-classification`, `question-answering`, `summarization`, `translation`, `text-generation`, `image-classification`, `automatic-speech-recognition`

---

### Loading Models & Tokenizers Directly

For more control, load the tokenizer and model separately:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_name = "distilbert-base-uncased-finetuned-sst-2-english"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Tokenize input
inputs = tokenizer("Transformers are amazing!", return_tensors="pt", truncation=True)
# inputs = {'input_ids': tensor([[...]]), 'attention_mask': tensor([[...]])}

# Run inference
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = logits.argmax(dim=-1).item()

print(model.config.id2label[predicted_class])  # 'POSITIVE'
```

---

### The `Auto` Classes

Always prefer `Auto*` classes — they automatically pick the right architecture:

| Class | Use Case |
|---|---|
| `AutoTokenizer` | Tokenization for any model |
| `AutoModel` | Base model (raw hidden states) |
| `AutoModelForSequenceClassification` | Text classification |
| `AutoModelForTokenClassification` | NER, POS tagging |
| `AutoModelForQuestionAnswering` | QA tasks |
| `AutoModelForCausalLM` | Text generation (GPT-style) |
| `AutoModelForSeq2SeqLM` | Translation, summarization (T5/BART) |

---

### Tokenizers in Depth

```python
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Batch encoding with padding/truncation
encoded = tokenizer(
    ["Hello world", "Transformers are great for NLP"],
    padding=True,        # Pad to longest in batch
    truncation=True,     # Truncate to model max length
    max_length=128,
    return_tensors="pt"  # "pt" = PyTorch, "tf" = TensorFlow, "np" = NumPy
)

# Decode tokens back to text
tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])
text = tokenizer.decode(encoded["input_ids"][0], skip_special_tokens=True)
```

---

### Fine-Tuning with the `Trainer` API

```python
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset

# Load data
dataset = load_dataset("imdb")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding=True)

tokenized = dataset.map(tokenize, batched=True)

# Load model (num_labels = number of output classes)
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2
)

# Training config
args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
)

trainer.train()
trainer.save_model("./my-fine-tuned-model")
```

---

### Saving & Loading Your Own Models

```python
# Save
model.save_pretrained("./my-model")
tokenizer.save_pretrained("./my-model")

# Load back
model = AutoModelForSequenceClassification.from_pretrained("./my-model")
tokenizer = AutoTokenizer.from_pretrained("./my-model")

# Push to Hugging Face Hub
model.push_to_hub("your-username/my-awesome-model")
tokenizer.push_to_hub("your-username/my-awesome-model")
```

---

### Running on GPU

```python
device = "cuda" if torch.cuda.is_available() else "cpu"

model = model.to(device)
inputs = {k: v.to(device) for k, v in inputs.items()}

# Or directly in pipeline:
pipe = pipeline("text-generation", model="gpt2", device=0)  # device=0 for first GPU
```

---

### Key Tips

- **Use `AutoTokenizer` / `AutoModel`** — don't hardcode architectures, it's more portable.
- **`from_pretrained` caches models** locally at `~/.cache/huggingface/` by default.
- **Half precision for speed:** `model.half()` or load with `torch_dtype=torch.float16` on GPU.
- **Browse models** at [huggingface.co/models](https://huggingface.co/models) — filter by task, language, size.
