from pathlib import Path

from datasets import load_dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parent.parent
samples_file = str(ROOT / "data" / "labeled_samples_800.json")
training_output_dir = str(ROOT / "training-output")
logs_dir = str(ROOT / "logs")
model_dir = ROOT / "models" / "ad-classifier"
model_dir.mkdir(parents=True, exist_ok=True)

# Load JSON dataset
dataset = load_dataset("json", data_files=samples_file, split="train")

# Split into train/test
dataset = dataset.train_test_split(test_size=0.2)

# Load tokenizer and model
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# Tokenize the text
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Define evaluation metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# Training configuration
training_args = TrainingArguments(
    output_dir=training_output_dir,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir=logs_dir,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Train!
trainer.train()

# Save final model
model.save_pretrained(model_dir)
tokenizer.save_pretrained(model_dir)
