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

# os.makedirs('./cats', exist_ok=True)
# os.makedirs('./dogs', exist_ok=True)

# # for file in os.listdir(dir):
# #     file_path = os.path.join(dir, file)
# #     if 'cat' in file.lower():
# #         shutil.copy(file_path, f"./cats/{file}")
# #     if 'dog' in file.lower():
# #         shutil.copy(file_path, f"./dogs/{file}")
        
# b)        
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
model = ImageClassifier().to(device)

criterion = nn.CrossEntropyLoss() # Definiuje funkcję błędu. Mierzy, jak bardzo przewidywania modelu mijają się z rzeczywistymi etykietami.
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Tworzy optymalizator Adama, który na podstawie błędu (obliczonego wyżej)
                                                            # będzie modyfikował wagi sieci
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


epochs = 3
train_losses = []
train_accs = []
val_losses = []
val_accs = []

for epoch in range(epochs):
    model.train()
    train_loss = 0
    train_acc = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad() # czyści gradienty
        outputs = model(images) # forward, przerzuca dane przez neruony
        loss = criterion(outputs, labels) # oblicza błąd (porównuje outputs z labels)
        loss.backward() # propagacja wstecz
        optimizer.step() # aktualizuje wagi sieci

        train_loss += loss.item() # sumuje błędy paczek do licznika epoki
        train_acc += (outputs.argmax(1) == labels).sum().item()
        
    train_losses.append(train_loss/len(train_loader)) # tworzy listę błędów epok
    train_accs.append(train_acc/len(train_dataset))

    # walidacja
    model.eval()
    val_loss = 0
    val_acc = 0

    with torch.no_grad(): # wyłącza śledzenie gradientów
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
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
plt.savefig("zad2_loss.png")
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
plt.savefig("zad2_acc.png")
plt.show()


wrong = []

model.eval()
with torch.no_grad():
    for images, labels in val_loader:
        outputs = model(images.to(device))
        preds = torch.argmax(outputs, dim=1).cpu()

        for i in range(len(preds)):
            if preds[i] != labels[i]:
                wrong.append((images[i], preds[i], labels[i]))

print("Liczba błędów:", len(wrong))


""" dropout
Epoch: 0
   - Train: acc=1069 (54.74)%; loss=17.885
   - Val: acc=289 (59.10)%; loss=11.624
Epoch: 1
   - Train: acc=1140 (58.37)%; loss=20.434
   - Val: acc=265 (54.19)%; loss=32.967
Epoch: 2
   - Train: acc=1131 (57.91)%; loss=31.200
   - Val: acc=258 (52.76)%; loss=18.750
Epoch: 3
   - Train: acc=1189 (60.88)%; loss=32.942
   - Val: acc=287 (58.69)%; loss=13.786
"""



# opt: adam; nn>ReLu
"""
Epoch: 0
   - Train: acc=1094 (56.02)%; loss=10.803
   - Val: acc=268 (54.81)%; loss=13.266
Epoch: 1
   - Train: acc=1166 (59.70)%; loss=12.336
   - Val: acc=280 (57.26)%; loss=11.713
Epoch: 2
   - Train: acc=1203 (61.60)%; loss=11.872
   - Val: acc=295 (60.33)%; loss=8.597
Liczba błędów: 194

"""
# opt: SGD; nn.ReLu
"""
Epoch: 0
   - Train: acc=1139 (58.32)%; loss=12.103
   - Val: acc=281 (57.46)%; loss=15.203
Epoch: 1
   - Train: acc=1182 (60.52)%; loss=16.674
   - Val: acc=288 (58.90)%; loss=19.009
Epoch: 2
   - Train: acc=1246 (63.80)%; loss=12.811
   - Val: acc=280 (57.26)%; loss=21.163
Liczba błędów: 209
"""

# opt: Adam; nn.Sigmoid
"""
Epoch: 0
   - Train: acc=1113 (56.99)%; loss=10.299
   - Val: acc=256 (52.35)%; loss=15.719
Epoch: 1
   - Train: acc=1128 (57.76)%; loss=12.887
   - Val: acc=285 (58.28)%; loss=10.738
Epoch: 2
   - Train: acc=1200 (61.44)%; loss=7.017
   - Val: acc=285 (58.28)%; loss=7.805
Liczba błędów: 204
"""

# opt: SGD; nn:Sigmoid
"""
Epoch: 0
   - Train: acc=1078 (55.20)%; loss=10.628
   - Val: acc=235 (48.06)%; loss=26.758
Epoch: 1
   - Train: acc=1127 (57.71)%; loss=13.419
   - Val: acc=262 (53.58)%; loss=31.993
Epoch: 2
   - Train: acc=1114 (57.04)%; loss=20.746
   - Val: acc=289 (59.10)%; loss=22.521
Liczba błędów: 200
"""

# opt: Adam; nn.ReLu => nn.Sigmoid => nn.ReLu
"""
Epoch: 0
   - Train: acc=1057 (54.12)%; loss=11.129
   - Val: acc=289 (59.10)%; loss=11.290
Epoch: 1
   - Train: acc=1106 (56.63)%; loss=12.858
   - Val: acc=292 (59.71)%; loss=11.250
Epoch: 2
   - Train: acc=1247 (63.85)%; loss=8.793
   - Val: acc=289 (59.10)%; loss=10.297
Liczba błędów: 200
"""

# opt: SGD; nn.Sigmoid => nn.ReLu => nn.Sigmoid
"""
Epoch: 0
   - Train: acc=1057 (54.12)%; loss=26.023
   - Val: acc=251 (51.33)%; loss=12.199
Epoch: 1
   - Train: acc=1052 (53.87)%; loss=37.533
   - Val: acc=259 (52.97)%; loss=11.751
Epoch: 2
   - Train: acc=1085 (55.56)%; loss=35.977
   - Val: acc=292 (59.71)%; loss=11.470
"""


# cm = confusion_matrix()


model.eval()
val_labels = []
pred_labels =[]

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        outputs = model(images)
        pred = outputs.argmax(1).cpu().numpy()
        
        pred_labels.extend(pred)
        val_labels.extend(labels)


cm = confusion_matrix(val_labels, pred_labels)
cm_df = pd.DataFrame(cm, index = ['kot', 'pies'], columns = ['kot', 'pies'])
print(cm_df)

# Najlepszy dropout + SGD + relu > sig > relu







