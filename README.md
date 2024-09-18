# Medical Assistant Chatbot

This project involves building a Medical Assistant Chatbot using a fine-tuned Large Language Model (Flan-T5) and deploying it via FastAPI and Gradio.

## Directory Structure
"""
backend/
├── dataset/
├── models/
├── scripts/
│   ├── prepare_dataset.py
│   ├── train_model.py
│   ├── train_lora.py
├── app/
│   ├── main.py
├── requirements.txt
├── README.md
"""
## Setup Instructions

1. Clone the repository.

2. Set up a virtual environment:

   python3 -m venv venv
   source venv/bin/activate

3. Install dependencies:

    pip install -r requirements.txt

4. Prepare the dataset:

    python3 scripts/prepare_dataset.py

5. Train the model:

    python3 scripts/train_lora.py

6. Run the FastAPI server:

    uvicorn app.main:app --reload

7. Project Details

- Dataset: ruslanmv/ai-medical-chatbot
- Model: google/flan-t5-small
- Frameworks: FastAPI, Gradio, Hugging Face Transformers
- LoRA: Low-Rank Adaptation for efficient fine-tuning.

8. Future Enhancements

- Continuous model retraining
- Scalability with Kubernetes

---
