from datasets import load_dataset
from transformers import AutoTokenizer

def preprocess_data():
    # Load dataset
    # dataset = load_dataset('eswardivi/medical_qa')
    dataset = load_dataset('sunilghanchi/medical-qna-3k')
    
    # Split the dataset into train and validation sets (90% train, 10% validation)
    dataset = dataset['train'].train_test_split(test_size=0.1)

    # Load tokenizer for Flan-T5
    model_name = 'google/flan-t5-small'
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Preprocess the data (tokenize) - Tokenize Patient as input and Doctor as target
    def tokenize_function(examples):
        # Tokenize the instruction 
        inputs = [f"Question: {instruction}" for instruction in examples['Question']]
        # Tokenize the output  (labels)
        targets = [f"Answer: {output}" for output in examples['Answer']]

        model_inputs = tokenizer(inputs, padding='max_length', truncation=True)
        labels = tokenizer(targets, padding='max_length', truncation=True)

        # Replace padding token id's in labels by -100 so they are ignored by the loss function
        labels["input_ids"] = [
            [(label if label != tokenizer.pad_token_id else -100) for label in labels_ids]
            for labels_ids in labels["input_ids"]
        ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # Apply the tokenization function to the dataset
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # Save the tokenized dataset to disk
    tokenized_dataset.save_to_disk("/home/project/BE/dataset/tokenized_data_medical-qna-3k")

if __name__ == "__main__":
    preprocess_data()
