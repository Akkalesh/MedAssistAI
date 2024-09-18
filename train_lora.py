from peft import get_peft_model, LoraConfig
from transformers import Trainer, TrainingArguments, T5ForConditionalGeneration, AutoTokenizer
from datasets import load_from_disk

# Load tokenized dataset
tokenized_data = load_from_disk("/home/project/BE/dataset/tokenized_data_medical-qna-3k")

# Load base model and tokenizer
model_name = 'google/flan-t5-small'  
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# LoRA configuration
lora_config = LoraConfig(
    r=8, 
    lora_alpha=32, 
    target_modules=["q", "v"], 
    lora_dropout=0.1, 
    bias="none"
)

# Apply LoRA
model = get_peft_model(model, lora_config)

# Training arguments
training_args = TrainingArguments(
    output_dir='/home/project/BE/models/lora_fine_tuned_model_medical-qna-3k',
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    save_steps=10_000,
    logging_dir='/home/project/BE/logs',
    eval_strategy="epoch",
    logging_strategy="steps",
    logging_steps=500,
    save_total_limit=2,
    learning_rate=5e-5,
)

# Initialize Trainer with the LoRA model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data['train'],
    eval_dataset=tokenized_data['test'],  # Validation set
)

# Train LoRA fine-tuned model
if __name__ == "__main__":
    trainer.train()
    trainer.save_model("/home/project/BE/models/lora_fine_tuned_model_medical-qna-3k")
