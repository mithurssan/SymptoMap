import numpy as np
import pickle
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score, f1_score, precision_score, classification_report

#Loading the data
with open("bert_splits.json", "r") as f:
    splits = json.load(f)
    
X_train = splits["X_train"]
X_val = splits["X_val"]
X_test = splits["X_test"]

y_train = np.load("y_train.npy")
y_val = np.load("y_val.npy")
y_test = np.load("y_test.npy")

numb_classes = len(np.unique(y_train))
print(f"Classes: {numb_classes} Train: {len(X_train)} Val: {len(X_val)} Test: {len(X_test)}")

#Pre-processing done within initialisation of BioClinicalBERT
#Tokeniser

model_name = "emilyalsentzer/Bio_ClinicalBERT"
print(f"Loading tokeniser from {model_name}.")
tokeniser = AutoTokenizer.from_pretrained(model_name)

print("Tokenising dataset...")

def tokenise(texts):
    return tokeniser(
        texts,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    )

train_enc = tokenise(X_train)
val_enc = tokenise(X_val)
test_enc = tokenise(X_test)

#Converting to TensorDataset
train_dataset = TensorDataset(
    train_enc["input_ids"],
    train_enc["attention_mask"],
    torch.tensor(y_train, dtype=torch.long)
)

val_dataset = TensorDataset(
    val_enc["input_ids"],
    val_enc["attention_mask"],
    torch.tensor(y_val, dtype=torch.long)
)

test_dataset = TensorDataset(
    test_enc["input_ids"],
    test_enc["attention_mask"],
    torch.tensor(y_test, dtype=torch.long)
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

print("Tokenising complete")

#Model
class BioClinicalBERT(nn.Module):
    def __init__(self, model_name, num_classes, dropout=0.3):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        return self.fc(cls_output)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")

print("Loading BioClinicalBERT weights")
model = BioClinicalBERT(model_name, numb_classes).to(device)

optimiser = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
loss_fn = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode="max", patience=2, factor=0.5)

#Training
epochs = 10
best_val_f1 = 0.0
count = 0
early_stop = 4

print("Training started")

for epoch in range(epochs):
    model.train()
    total_loss = 0
    
    for input_ids, attention_mask, labels in train_loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        
        optimiser.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = loss_fn(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) #exploding gradient prevention
        optimiser.step()
        total_loss += loss.item()

    #Validation
    model.eval()
    preds_val, labels_val = [], []
    with torch.no_grad():
        for input_ids, attention_mask, labels in val_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            preds = model(input_ids, attention_mask).argmax(dim=1).cpu()
            preds_val.extend(preds.numpy())
            labels_val.extend(labels.numpy())
        
    val_acc = accuracy_score(labels_val, preds_val)
    val_f1 = f1_score(labels_val, preds_val, average="macro")
    scheduler.step(val_f1)
    
    print(f"Epoch {epoch+1:02d}/{epochs}" f" Loss: {total_loss/len(train_loader):.4f}" f" Val Acc: {val_acc:.4f} Val F1: {val_f1:.4f}" + (" <- best" if val_f1 > best_val_f1 else ""))
    
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), "bert_best.pt")
        count = 0
    else:
        count += 1
        if count >= early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break

#Test evaluation
print("Loading best model for the final evaluation")
model.load_state_dict(torch.load("bert_best.pt", map_location=device))
model.eval()

preds_test, labels_test = [], []
with torch.no_grad():
    for input_ids, attention_mask, labels in test_loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        preds = model(input_ids, attention_mask).argmax(dim=1).cpu()
        preds_test.extend(preds.numpy())
        labels_test.extend(labels.numpy())
        
with open ("label_encoder.pickle", "rb") as f:
    le = pickle.load(f)
    
print("-BIOCLINICAL BERT FINAL TEST RESULTS-")
print(f"Accuracy: {accuracy_score(labels_test, preds_test):.4f}")
print(f"F1: {f1_score(labels_test, preds_test, average="macro"):.4f}")
print(f"Precision: {precision_score(labels_test, preds_test, average="macro"):.4f}")
print("Full report:")
print(classification_report(labels_test, preds_test, target_names=le.classes_, digits=4))

np.save("bert_test_preds.npy", np.array(preds_test))
np.save("bert_test_labels.npy", np.array(labels_test))
print("Saved: bert_best.pt, bert_test_preds.npy, bert_test_labels.npy")


#External Heldout test
with open("bert_ext.json") as f:
    X_ext_text = json.load(f)["X_ext"]

y_ext = torch.tensor(np.load("y_ext.npy"), dtype=torch.long)

ext_encodings = tokeniser(
    X_ext_text,
    truncation=True, padding=True,
    max_length=128, return_tensors="pt"
)

ext_loader = DataLoader(
    TensorDataset(ext_encodings["input_ids"], ext_encodings["attention_mask"], y_ext),
    batch_size=32, shuffle=False
)

model.load_state_dict(torch.load("bert_best.pt", map_location=device))
model.eval()
preds_ext, labels_ext = [], []
with torch.no_grad():
    for input_ids, attention_mask, y_batch in ext_loader:
        outputs = model(input_ids.to(device), attention_mask.to(device))
        preds = outputs.argmax(dim=1).cpu()
        preds_ext.extend(preds.numpy())
        labels_ext.extend(y_batch.numpy())

print("-BIOCLINICALBERT EXTERNAL TEST RESULTS (Cross-LLM)-")
print(f"Accuracy:{accuracy_score(labels_ext, preds_ext):.4f}")
print(f"F1:{f1_score(labels_ext, preds_ext, average="macro"):.4f}")
print(f"Precision:{precision_score(labels_ext, preds_ext, average="macro"):.4f}")
print(classification_report(labels_ext, preds_ext, target_names=le.classes_, digits=4))

np.save("bert_ext_preds.npy", np.array(preds_ext))
np.save("bert_ext_labels.npy", np.array(labels_ext))
