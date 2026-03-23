import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler  
import matplotlib.pyplot as plt
import seaborn as sbs


df = pd.read_csv("iris_big.csv")

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# kodowanie klas
le = LabelEncoder()
y = le.fit_transform(y)

(X_train, X_test, y_train, y_test) = train_test_split(X, y, train_size=0.7, random_state=300869)

zScore = StandardScaler()
X_train_zScore = zScore.fit_transform(X_train)
X_test_zScore = zScore.transform(X_test)


# # UŻYCIE WBUDOWANEGO TensorDataset ZAMIAST WŁASNEJ KLASY

# konwersja na tensory (tablicy jak numpy ale można robić obliczenia na CUDA)
# X_train_tensor = torch.tensor(X_train_zScore, dtype=torch.float32)
# y_train_tensor = torch.tensor(y_train, dtype=torch.long)
# X_test_tensor = torch.tensor(X_test_zScore, dtype=torch.float32)
# y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
# val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

class IrisDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]
    
train_dataset = IrisDataset(X_train_zScore, y_train)
test_dataset = IrisDataset(X_test_zScore, y_test)

batch_size = 64

train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)



device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, 3)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# pod wykresy
train_losses = []
test_losses = []
train_accs = []
test_accs = []

def train(dataloader, model, loss_fn, optimizer):
    
    train_loss, train_acc = 0, 0
    batches = len(dataloader)
    
    size = len(dataloader.dataset)
    model.train() # ustawia model w tryb treningowy
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()         # liczy gradienty
        optimizer.step()        # aktualizuje wagi 
        optimizer.zero_grad()   # zeruje gradienty bo pytorch je sumuje 
        
        train_loss += loss.item()
        train_acc += (pred.argmax(1) == y).type(torch.float).sum().item()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            
    return train_loss / batches, train_acc / size
            
            
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval() # ustawia model w tryb ewaluacji
    test_loss, correct = 0, 0
    predictions = []
    true_values = []
    
    with torch.no_grad(): # wyłącza obliczanie gradientów
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            
            predictions.extend(pred.argmax(1).cpu().numpy())
            true_values.extend(y.cpu().numpy())
            
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
    return test_loss, correct, predictions
    
epochs = 500
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loss, train_acc = train(train_dataloader, model, loss_fn, optimizer)
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    
    test_loss, test_acc, predictions = test(test_dataloader, model, loss_fn)
    test_losses.append(test_loss)
    test_accs.append(test_acc)
    
    if (t+1) % 10 == 0:
            print(f"Epoch {t+1}/{epochs} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")
    

print("Done!")

# tutaj strata jest liczona do prawdopodobieńtwa 1.0
# mamy na wyjściu np. 1.2, to istnieje jakaś strata

epochs_range = range(1, epochs+1)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(epochs_range, train_losses, label="Trening")
ax1.plot(epochs_range, test_losses, label="Test")
ax1.set(xlabel="epoki", ylabel="błąd", title="Wykres Loss")
ax1.legend()
ax1.grid(True)

ax2.plot(epochs_range, train_accs, label="Trening")
ax2.plot(epochs_range, test_accs, label="Test")
ax2.set(xlabel="epoki", ylabel="Dokładność", title="Wykres Acc")
ax2.legend()
ax2.grid(True)

plt.show()


# macierz i acc score
print("\n\n====== Koncowe statystyki ======")
acc = accuracy_score(y_test, predictions)
print(f"Acc score: {round(acc*100, 4)}%")

confMatrix = confusion_matrix(y_test, predictions)
confMatrix_df = pd.DataFrame(confMatrix, index = ['setosa', 'virginica', 'versicolor'], columns= ['setosa', 'virginica', 'versicolor'])
print(confMatrix_df)