import glob
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split

from nltk_preprocess import preprocess_text

train_neg_files = glob.glob("aclImdb/train/neg/*.txt")[:5000]
train_pos_files = glob.glob("aclImdb/train/pos/*.txt")[:5000]
test_neg_files = glob.glob("aclImdb/test/neg/*.txt")[:5000]
test_pos_files = glob.glob("aclImdb/test/pos/*.txt")[:5000]

def read_file(filename):
    with open(filename, 'r', encoding="utf-8") as fd:
        return fd.read()

print("Preprocessing reviews")
train_reviews = []
test_reviews = []

for text in map(preprocess_text,
                map(read_file, train_neg_files)):
    train_reviews.append((text, 0))

for text in map(preprocess_text,
                map(read_file, train_pos_files)):
    train_reviews.append((text, 1))

for text in map(preprocess_text,
                map(read_file, test_neg_files)):
    test_reviews.append((text, 0))

for text in map(preprocess_text,
                map(read_file, test_pos_files)):
    test_reviews.append((text, 1))

import random
random.shuffle(train_reviews)
random.shuffle(test_reviews)


X_train = [r[0] for r in train_reviews]
y_train = [r[1] for r in train_reviews]

X_test = [r[0] for r in test_reviews]
y_test = [r[1] for r in test_reviews]

# Load your preprocessed data and labels
#data = np.array(preprocessed_data)
#labels = np.array(labels)

# Split data into training and testing sets
#X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Load the tokenizer and the pre-trained DistilBERT model
print("loading the tokenizer")
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

# Prepare the input data
print("preparing the input data")
train_encodings = tokenizer(X_train, truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(X_test, truncation=True, padding=True, max_length=128)

# Create the input tensors
print("creating input tensors")
train_inputs = torch.tensor(train_encodings['input_ids'])
train_masks = torch.tensor(train_encodings['attention_mask'])
train_labels = torch.tensor(y_train, dtype=torch.long)

test_inputs = torch.tensor(test_encodings['input_ids'])
test_masks = torch.tensor(test_encodings['attention_mask'])
test_labels = torch.tensor(y_test, dtype=torch.long)

# Create data loaders
print("creating data loaders")
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_dataloader = DataLoader(train_data, batch_size=16, shuffle=True)

test_data = TensorDataset(test_inputs, test_masks, test_labels)
test_dataloader = DataLoader(test_data, batch_size=16, shuffle=False)

# Set up the optimizer
print("setting up the optimizer")
optimizer = AdamW(model.parameters(), lr=2e-5)

# Training loop
epochs = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(epochs):
    model.train()
    total_train_loss = 0

    for batch in train_dataloader:
        input_ids, attention_mask, labels = tuple(t.to(device) for t in batch)
        model.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        total_train_loss += loss.item()
        loss.backward()
        optimizer.step()

    print(f'Epoch: {epoch + 1}, Loss: {total_train_loss / len(train_dataloader)}')

print("Saving the model&tokenizer")
tokenizer.save_pretrained("sentiment-analysis-tokenizer")
model.save_pretrained("sentiment-analysis-model")

# Evaluation
model.eval()
correct = 0
total = 0

for batch in test_dataloader:
    input_ids, attention_mask, labels = tuple(t.to(device) for t in batch)
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs[0]

    _, predicted = torch.max(logits, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f'Test accuracy: {accuracy}')
