import torch
import os 
import shutil
from torch.utils.data import DataLoader, random_split   
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd
import torchvision.models as models

dir = 'dogs-cats-mini'


# f)       
img_w = 128
img_h = 128
img_size = (img_w, img_h)

data_transform = transforms.Compose([
    transforms.Resize(size=img_size), # resize do danych wymiarów
    #transforms.RandomHorizontalFlip(p=0.5), # p - prawdopodobieństwo obrócenia zdjęcia
    transforms.ToTensor(), # zmiana zdjęcia na torch.Tensor, zmienia też wartości pixeli z 0-255 do 0.0-1.0
    transforms.Normalize([0.34, 0.5, 0.4], [0.5, 0.2, 0.43])
])

# automatycznie klasyfikuje na koty,psy = 0,1
dataset = datasets.ImageFolder(root="./data", transform=data_transform, target_transform=None)

class_dict = dataset.class_to_idx
print("Class names as a dict: ",class_dict)

train_size = int(0.8 * len(dataset))
val_size = len(dataset  ) - train_size


train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)


print(f"Właściwości danych:\n  - All: {len(dataset)}\n  - Train: {len(train_dataset)} => {len(train_loader)}\n  - Val: {len(val_dataset)} => {len(val_loader)}")

# c)
# # Creating a CNN-based image classifier.
class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layer_1 = nn.Sequential(
          nn.Conv2d(3, 64, 3, padding=1),
          nn.ReLU(),
        #   nn.Sigmoid(),
          nn.BatchNorm2d(64),
          nn.MaxPool2d(2))
        
        self.conv_layer_2 = nn.Sequential(
          nn.Conv2d(64, 512, 3, padding=1),
        #   nn.ReLU(),
          nn.Sigmoid(),
          nn.BatchNorm2d(512),
          nn.MaxPool2d(2))
        
        self.conv_layer_3 = nn.Sequential(
          nn.Conv2d(512, 512, kernel_size=3, padding=1),
          nn.ReLU(),
        #   nn.Sigmoid(),
          nn.Dropout(p=0.5),
          nn.BatchNorm2d(512),
          nn.MaxPool2d(2)) 
        
        self.classifier = nn.Sequential(
          nn.Flatten(),
          nn.Linear(in_features=512*16*16, out_features=2))
        
    def forward(self, x: torch.Tensor):
        x = self.conv_layer_1(x)
        x = self.conv_layer_2(x)
        x = self.conv_layer_3(x)
        x = self.classifier(x)
        return x
# Instantiate an object.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

criterion = nn.CrossEntropyLoss() # Definiuje funkcję błędu. Mierzy, jak bardzo przewidywania modelu mijają się z rzeczywistymi etykietami.

model2 = models.resnet18(pretrained=True)

# wyłączamy p - bias, wagi, by nie usunąć wytrenowanych
for p in model2.parameters():
    p.requires_grad = False
    
# zmiana wyjść modelu na 2 klasy
model2.fc = nn.Linear(model2.fc.in_features, 2)

model2 = model2.to(device)

optimizer = torch.optim.Adam(model2.fc.parameters(), lr=0.001)




epochs = 3
train_losses = []
train_accs = []
val_losses = []
val_accs = []

for epoch in range(epochs):
    model2.train()
    train_loss = 0
    train_acc = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad() # czyści gradienty
        outputs = model2(images) # forward, przerzuca dane przez neruony
        loss = criterion(outputs, labels) # oblicza błąd (porównuje outputs z labels)
        loss.backward() # propagacja wstecz
        optimizer.step() # aktualizuje wagi sieci

        train_loss += loss.item() # sumuje błędy paczek do licznika epoki
        train_acc += (outputs.argmax(1) == labels).sum().item()
        
    train_losses.append(train_loss/len(train_loader)) # tworzy listę błędów epok
    train_accs.append(train_acc/len(train_dataset))

    # walidacja
    model2.eval()
    val_loss = 0
    val_acc = 0

    with torch.no_grad(): # wyłącza śledzenie gradientów
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model2(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            val_acc += (outputs.argmax(1) == labels).sum().item()

    val_losses.append(val_loss/len(val_loader))
    val_accs.append(val_acc/len(val_dataset))

    # print(f"Epoch {epoch}: train={train_loss:.3f}, val={val_loss:.3f}")
    print(f"Epoch: {epoch}\n",
          f"  - Train: acc={train_acc} ({train_acc/len(train_dataset)*100:.2f})%; loss={train_loss/len(train_loader):.3f}\n"
          f"   - Val: acc={val_acc} ({val_acc/len(val_dataset)*100:.2f})%; loss={val_loss/len(val_loader):.3f}"
          )

epochs_range = range(1, epochs + 1)

# loss
plt.figure() 
plt.plot(epochs_range, train_losses, label="train")
plt.plot(epochs_range, val_losses, label="val")
plt.grid(True)
plt.xticks(epochs_range)
plt.xlabel("epoki")
plt.ylabel("loss (średnia epoki / batch_am)")
plt.legend()
plt.savefig("zad2_pkt-f_loss.png")
plt.show()

# acc
plt.figure() 
plt.plot(epochs_range, train_accs, label="train")
plt.plot(epochs_range, val_accs, label="val")
plt.grid(True)
plt.xticks(epochs_range)
plt.xlabel("epoki")
plt.ylabel("accuracy")
plt.legend()
plt.savefig("zad2_pkt-f_acc.png")
plt.show()


wrong = []

model2.eval()
with torch.no_grad():
    for images, labels in val_loader:
        outputs = model2(images.to(device))
        preds = torch.argmax(outputs, dim=1).cpu()

        for i in range(len(preds)):
            if preds[i] != labels[i]:
                wrong.append((images[i], preds[i], labels[i]))

print("Liczba błędów:", len(wrong))




"""
Epoch: 0
   - Train: acc=1395 (71.43)%; loss=0.568
   - Val: acc=360 (73.62)%; loss=0.517
Epoch: 1
   - Train: acc=1520 (77.83)%; loss=0.448
   - Val: acc=369 (75.46)%; loss=0.522
Epoch: 2
   - Train: acc=1561 (79.93)%; loss=0.421
   - Val: acc=361 (73.82)%; loss=0.521
Liczba błędów: 128
"""