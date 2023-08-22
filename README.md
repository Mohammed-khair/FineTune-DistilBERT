# Fine-Tuning DistilBERT for NSFW Classification

This Jupyter Notebook demonstrates how to fine-tune the DistilBERT model for the classification of NSFW (Not Safe for Work) prompts. The notebook includes steps for installing required libraries, importing data, preprocessing, tokenization, creating the model, and training it.

## Table of Contents
1. [Introduction](#introduction)
2. [Installation and Libraries](#installation-and-libraries)
3. [Data Import](#data-import)
4. [Data Preprocessing](#data-preprocessing)
5. [Tokenization](#tokenization)
6. [Data Splitting](#data-splitting)
7. [Model Creation](#model-creation)
8. [Model Training](#model-training)
9. [Saving the Model](#saving-the-model)

## Introduction <a name="introduction"></a>
This notebook focuses on fine-tuning the DistilBERT model to classify NSFW prompts. We will use the HuggingFace `transformers` library, TensorFlow, and Scikit-learn for various tasks such as data handling, preprocessing, tokenization, and model training.

## Installation and Libraries <a name="installation-and-libraries"></a>
The first step is to install and import the necessary libraries:

```python
import tensorflow as tf
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
from sklearn.model_selection import train_test_split
import numpy as np
```

## Data Import <a name="data-import"></a>
We will import the training data from the HuggingFace dataset:

```python
from datasets import load_dataset

dataset = load_dataset("thefcraft/civitai-stable-diffusion-337k")
data = dataset["train"]
```

## Data Preprocessing <a name="data-preprocessing"></a>
Extract the prompts, negative prompts, and labels from the data:

```python
prompts = [d['prompt'] for d in data]
neg_prompts = [d['negativePrompt'] for d in data]
labels = [d['nsfw'] for d in data]
```

## Tokenization <a name="tokenization"></a>
Combine prompts and negative prompts and tokenize them using the DistilBERT tokenizer:

```python
prompts_combined = ["Positive prompt: " + pos_prompt + ". Negative prompt: " + neg_prompt for pos_prompt, neg_prompt in zip(prompts, neg_prompts)]

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
inputs = tokenizer(prompts_combined, padding=True, truncation=True, return_tensors="tf")
```

## Data Splitting <a name="data-splitting"></a>
Split the data into training and testing sets:

```python
input_ids = inputs['input_ids'].numpy()
attention_mask = inputs['attention_mask'].numpy()
labels = [int(x) for x in labels]

input_ids_train, input_ids_test, attention_mask_train, attention_mask_test, labels_train, labels_test = train_test_split(
    input_ids, attention_mask, labels, test_size=0.2, random_state=42
)
```

## Model Creation <a name="model-creation"></a>
Create the DistilBERT-based classification model:

```python
model = TFDistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=1)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5), loss='binary_crossentropy', metrics=['accuracy'])
```

## Model Training <a name="model-training"></a>
Train the model using the training data:

```python
history = model.fit(
    {'input_ids': input_ids_train_tensor, 'attention_mask': attention_mask_train_tensor},
    train_labels_tensor,
    epochs=3,
    batch_size=8,
    validation_split=0.2
)
```

## Saving the Model <a name="saving-the-model"></a>
Save the trained model for future use:

```python
model.save('models/model.keras')
```

This notebook demonstrates the process of fine-tuning the DistilBERT model for the classification of NSFW prompts. It covers data preprocessing, tokenization, model creation, training, and model saving. The trained model can then be utilized for NSFW classification tasks.
