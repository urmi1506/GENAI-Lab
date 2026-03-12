
#import libraries
import torch
from datasets import Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

# Load pre-trained GPT-2 model and tokenizer
model_name = "gpt2"

tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# GPT2 has no default padding token
tokenizer.pad_token = tokenizer.eos_token

# Sample creative story dataset
stories = [
    "Once upon a time, a curious robot discovered a hidden library beneath the city.",
    "In a quiet village, a young girl found a magical key that opened doors to different worlds.",
    "A lonely astronaut drifting through space heard a mysterious signal from an unknown planet.",
    "Deep in the enchanted forest, animals gathered every night to tell stories under the moonlight.",
    "A time traveler accidentally changed history and had to fix the future before it disappeared."
]

dataset = Dataset.from_dict({"text": stories})

print(dataset)

# Tokenization function
def tokenize_function(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=64
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True)

#Data Collator - prepares data for language model training.
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False   # GPT models use causal language modeling
)

#Training Configuration
training_args = TrainingArguments(
    output_dir="./gpt2-story",
    # overwrite_output_dir=True, # This parameter is causing the error
    num_train_epochs=5,
    per_device_train_batch_size=2,
    save_steps=500,
    save_total_limit=2,
    logging_steps=100
)

#Trainer Setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

#Fine-Tune GPT-2
trainer.train()

# Function to generate stories
def generate_story(prompt):

    inputs = tokenizer(prompt, return_tensors="pt")

    output = model.generate(
        **inputs,
        max_length=120,
        num_return_sequences=1,
        temperature=0.9,
        do_sample=True,
        top_k=50
    )

    story = tokenizer.decode(output[0], skip_special_tokens=True)
    print(story)

# Example prompt
generate_story("Once upon a time in a Pune")

# Interactive Story Generator
while True:

    prompt = input("Enter story prompt (or type exit): ")

    if prompt.lower() == "exit":
        break

    generate_story(prompt)
