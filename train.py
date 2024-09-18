import torch
from transformers import Trainer, TrainingArguments, AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_from_disk

tokenized_data = load_from_disk("/home/project/BE/dataset/tokenized_data")

model_name = 'google/flan-t5-small'
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


# Training setup
training_args = TrainingArguments(
    output_dir="/home/project/BE/models",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    logging_dir='/home/project/BE/logs',
    save_steps=10_000,
    evaluation_strategy="epoch",
    logging_strategy="steps",
    logging_steps=500,
    save_total_limit=2,
    learning_rate=5e-5,
)

# Initiate the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["test"]
)

if __name__ == "__main__":
    trainer.train()
    train.save_model("/home/project/BE/models/fine_tuned_model")
