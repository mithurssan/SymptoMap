import numpy as np
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, classification_report

#Loading the data
X_train = torch.tensor(np.load("X_train_tfidf.npy"), dtype=torch.float32)
X_val = torch.tensor(np.load("X_val_tfidf.npy"), dtype=torch.float32)
X_test = torch.tensor(np.load("X_test_tfidf.npy"), dtype=torch.float32)
y_train = torch.tensor(np.load("y_train.npy"), dtype=torch.long)
y_val = torch.tensor(np.load("y_val.npy"), dtype=torch.long)
y_test = torch.tensor(np.load("y_test.npy"), dtype=torch.long)

#adding dimension to fit expected input
X_train = X_train.unsqueeze(1)
X_val = X_val.unsqueeze(1)
X_test = X_test.unsqueeze(1)


numb_classes = len(torch.unique(y_train))
print(f"Classes: {numb_classes} Train: {len(X_train)} Test: {len(X_test)} Val: {len(X_val)}")

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32)


#Model
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=dropout if num_layers>1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:,-1,:])
        return self.fc(out)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")

model = BiLSTM(
    input_size=5000,
    hidden_size=128,
    num_layers=2,
    num_classes=numb_classes,
    dropout=0.3
).to(device)

loss_fn = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode="max", patience=3, factor=0.5) 

#Training
epochs = 30
best_val_f1 = 0.0
count = 0
early_stop = 7

print("Training Started.")

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optim.zero_grad()
        outputs = model(X_batch)
        loss = loss_fn(outputs, y_batch)
        loss.backward()
        optim.step()
        total_loss += loss.item()

#Validation
    model.eval()
    preds_val, labels_val = [], []
    with torch.no_grad(): #no gradient tracking
        for X_batch, y_batch in val_loader:
            preds = model(X_batch.to(device)).argmax(dim=1).cpu()
            preds_val.extend(preds.numpy())
            labels_val.extend(y_batch.numpy())
            
    val_acc = accuracy_score(labels_val, preds_val)
    val_f1 = f1_score(labels_val, preds_val, average="macro")
    scheduler.step(val_f1)
    
    print(f"Epoch {epoch+1:02d}/{epochs}" f" Loss: {total_loss/len(train_loader):.4f}" + (" <- best" if val_f1 > best_val_f1 else ""))
    
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), "bilstm_best.pt")
        count = 0
    else:
        count +=1
        if count >= early_stop:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
            
#Test evaluation
print("Loading best model for the final evaluation")
model.load_state_dict(torch.load("bilstm_best.pt", map_location=device))
model.eval()

preds_test, labels_test = [],[]
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        preds = model(X_batch.to(device)).argmax(dim=1).cpu()
        preds_test.extend(preds.numpy())
        labels_test.extend(y_batch.numpy())

with open("label_encoder.pickle", "rb") as f:
    le = pickle.load(f)

print("-BILSTM FINAL TEST RESULTS-")
print(f"Accuracy: {accuracy_score(labels_test, preds_test):.4f}")
print(f"F1: {f1_score(labels_test, preds_test, average="macro"):.4f}")
print(f"Precision: {precision_score(labels_test, preds_test, average="macro"):.4f}")
print("Full report:")
print(classification_report( labels_test, preds_test, target_names=le.classes_,digits=4))

np.save("bilstm_test_preds.npy", np.array(preds_test))
np.save("bilstm_test_labels.npy", np.array(labels_test))
print("Saved: bilstm_best.pt , bilstm_test_preds.npy, bilstm_test_labels.npy ")

#External Heldout test
X_ext = torch.tensor(np.load("X_ext_tfidf.npy"), dtype=torch.float32).unsqueeze(1)
y_ext = torch.tensor(np.load("y_ext.npy"), dtype=torch.long)

ext_loader = DataLoader(TensorDataset(X_ext, y_ext), batch_size=32, shuffle=False)
model.load_state_dict(torch.load("bilstm_best.pt", map_location = device))
model.eval()
preds_ext, labels_ext = [], []
with torch.no_grad():
    for X_batch, y_batch in ext_loader:
        preds = model(X_batch.to(device)).argmax(dim=1).cpu()
        preds_ext.extend(preds.numpy())
        labels_ext.extend(y_batch.numpy())

print("-BILSTM EXTERNAL TEST RESULTS (Cross-LLM)-")
print(f"Accuracy:{accuracy_score(labels_ext, preds_ext):.4f}")
print(f"F1:{f1_score(labels_ext, preds_ext, average="macro"):.4f}")
print(f"Precision:{precision_score(labels_ext, preds_ext, average="macro"):.4f}")
print(classification_report(labels_ext, preds_ext, target_names=le.classes_, digits=4))

np.save("bilstm_ext_preds.npy", np.array(preds_ext))
np.save("bilstm_ext_labels.npy", np.array(labels_ext))
