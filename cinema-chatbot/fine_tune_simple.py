from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

# 1. Charger les données (IMDB - classification de sentiments)
dataset = load_dataset("imdb")

# 2. Charger le tokenizer et le modèle DistilBERT
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Réduire la taille des données pour accélérer l'entraînement
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(500))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(500))

# 3. Charger DistilBERT pour la classification
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 4. Configurer l'entraînement
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=2,
    weight_decay=0.01,
    save_strategy="no",
)

# 5. Créer le Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
)

# 6. Lancer l'entraînement
trainer.train()

# 7. Sauvegarder le modèle et le tokenizer
model.save_pretrained("./distilbert_imdb_model", safe_serialization=False)
tokenizer.save_pretrained("./distilbert_imdb_model")
print("✅ Modèle et tokenizer sauvegardés dans ./distilbert_imdb_model")
