#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 12:10:53 2024

@author: farismismar
"""

import os
import glob
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import Dataset

# Ask: who is Muhammad?
# These parameters work fine.
mode = 'pretrained'
max_epochs = 10
batch_size = 8
context_window = 1024
temperature = None  # 1.2 # or None
seed = 42
learning_rate=2e-5

# Check if GPU is available
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
torch.backends.cudnn.enabled = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

torch.manual_seed(seed)

def read_pdfs(folder_path):
    import PyPDF2
    texts = []
    pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))
    for pdf_file in pdf_files:
        with open(pdf_file, "rb") as f:
            pdf_reader = PyPDF2.PdfReader(f)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            texts.append(text)
    return "\n".join(texts)


def read_text_files(folder_path):    
    texts = []
    txt_files = glob.glob(os.path.join(folder_path, "*.txt"))

    for txt_file in txt_files:
        if "output.txt" in txt_file:
            continue
        print(txt_file)
        with open(txt_file, "r", encoding='utf-8') as f:
            text = f.read()
            texts.append(text)

    texts = " ".join(texts)
    output_path = f"{folder_path}/output.txt"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(texts)
    f.close()
    
    return output_path


def preprocess_text(file_path, context_window):    
    with open(file_path, "r", encoding="utf-8") as f:
        data = f.read()
    
    # Split the data into chunks of context_window size
    examples = [data[i:i + context_window] for i in range(0, len(data), context_window)]
    
    # Return as a Hugging Face dataset
    return Dataset.from_dict({"text": examples})


# https://huggingface.co/docs/transformers/en/model_doc/gpt2
def train_model_pretrained_on_texts(text_file, output_dir="./fine_tuned_model"):
    global device, max_epochs, batch_size
    global context_window, learning_rate
    
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # Ensure padding token is set

    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)

    # Preprocess the text data
    dataset = preprocess_text(text_file, context_window)
    
    # Tokenize the dataset    
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=context_window)  # padding="max_length", 
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # GPT-2 does not use masked language modeling
    )
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=max_epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate, 
        save_steps=500,
        save_total_limit=2,
        prediction_loss_only=True,
        logging_dir="./logs",
        fp16=(device == 'cuda'),  # Enable mixed precision for GPUs
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset,
    )
    
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    return model, tokenizer


def train_model_scratch_on_texts(text_file, output_dir="./fine_tuned_model"):
    global device, max_epochs, batch_size
    global context_window, learning_rate
    
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # Ensure padding token is set

    # Initialize a new model configuration and model from scratch
    config = GPT2Config()
    model = GPT2LMHeadModel(config).to(device)

    # Preprocess the text data
    dataset = preprocess_text(text_file, context_window)
    
    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", max_length=context_window)
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # GPT-2 does not use masked language modeling
    )
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=max_epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        save_steps=500,
        save_total_limit=2,
        prediction_loss_only=True,
        logging_dir="./logs",
        fp16=(device == 'cuda'),  # Enable mixed precision for GPUs
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset,
    )
    
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    return model, tokenizer


def load_or_train_model(folder_path, model_name="gpt2", output_dir="./fine_tuned_model"):    
    global context_window
    global mode
    
    if os.path.exists(output_dir) and os.listdir(output_dir):
        print("Loading existing fine-tuned model...")
        model = GPT2LMHeadModel.from_pretrained(output_dir).to(device)
        tokenizer = GPT2Tokenizer.from_pretrained(output_dir)
    else:
        print(f"No fine-tuned model found.  Training {mode}...")
        text_file = read_text_files(folder_path)
        if mode == 'pretrained':
            model, tokenizer = train_model_pretrained_on_texts(text_file, output_dir)
        else:
            model, tokenizer = train_model_scratch_on_texts(text_file, output_dir)
        
    return model, tokenizer


def answer_questions(model, tokenizer, question):
    global device, context_window, temperature
    
    model.eval()  # Ensure the model is in evaluation mode
    inputs = tokenizer(question, return_tensors="pt", padding=True).to(device)

    outputs = model.generate(
        inputs["input_ids"],
        do_sample=(temperature is not None),
        temperature=temperature,
        attention_mask=inputs["attention_mask"],
        max_length=context_window,
        pad_token_id=tokenizer.eos_token_id
    )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Apply stop tokens
    stop_tokens = ['.', '?', '\n']
    for stop_token in stop_tokens:
        if stop_token in generated_text:
            generated_text = generated_text.split(stop_token)[0]
            break

    return generated_text


def build_llm():
    folder_path = "./documents"
    output_dir = "./fine_tuned_model"
    
    model, tokenizer = load_or_train_model(
        folder_path=folder_path,
        output_dir=output_dir
    )

    print("Ready.")
    while True:
        question = input("\nAsk a question (or type `exit' to quit): ")
        if question.lower() == "exit":
            print("Bye!")
            return 0
        answer = answer_questions(model, tokenizer, question)
        print(f"Answer: {answer}.")


def main():
    return build_llm()


if __name__ == "__main__":
    main()

