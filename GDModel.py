import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

dataDir = "dataset"
classes = ["openPalm", "fist", "wShape", "none"]
#mapping dictionary
labelToId = {"openPalm": 0, "fist": 1, "wShape": 2, "none": 3}
inDim = 42
batchSize = 32
lR = 1e-3
epochs = 20
seed = 42

torch.manual_seed(seed)

#dataframes
dfs = []
for c in classes:
    path = os.path.join(dataDir, f"{c}.csv")
    df = pd.read_csv(path)
    dfs.append(df)

#concat all dataframes into one
df = pd.concat(dfs, ignore_index=True)

#get the columns: sampleid, lable, x1, ....
cols = list(df.columns)
#from x1, x2, ....
featureCols = cols[2:] 

#convert the df to numpy
data = df[featureCols].to_numpy(dtype=np.float32)
labels = df["label"].map(labelToId).to_numpy(dtype=np.int64)

#split dataset: train & temp then split temp: val & test
dataTrain, dataTemp, labelTrain, labelTemp = train_test_split(
    data, labels, test_size=0.3, stratify=labels, random_state=seed
)

dataVal, dataTest, labelVal, labelTest = train_test_split(
    dataTemp, labelTemp, test_size=0.50, stratify=labelTemp, random_state=seed
)

#scale data
scaler = StandardScaler().fit(dataTrain)
dataTrain = scaler.transform(dataTrain).astype(np.float32)
dataVal = scaler.transform(dataVal).astype(np.float32)
dataTest = scaler.transform(dataTest).astype(np.float32)

#datasets/dataloaders
trainDS = torch.utils.data.TensorDataset(torch.from_numpy(dataTrain), torch.from_numpy(labelTrain))
valDS = torch.utils.data.TensorDataset(torch.from_numpy(dataVal),   torch.from_numpy(labelVal))
testDS = torch.utils.data.TensorDataset(torch.from_numpy(dataTest),  torch.from_numpy(labelTest))

trainDL = torch.utils.data.DataLoader(trainDS, batch_size=batchSize, shuffle=True, drop_last=False)
valDL = torch.utils.data.DataLoader(valDS,   batch_size=256, shuffle=False, drop_last=False)
testDL = torch.utils.data.DataLoader(testDS,  batch_size=256, shuffle=False, drop_last=False)

#model
class GDModel(nn.Module):
    def __init__(self, inDim=42, numClass=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inDim, 128), 
            nn.ReLU(),
            nn.Linear(128, 128), 
            nn.ReLU(),
            nn.Linear(128, numClass)
        )
    def forward(self, x): 
        return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GDModel().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=lR)

#evaluation function
def eval(valDL):
    model.eval()
    total, correct, lossSum = 0, 0, 0.0
    with torch.no_grad():
        for data, labels in valDL:
            data, labels = data.to(device), labels.to(device)
            pred = model(data)
            loss = criterion(pred, labels)
            lossSum += loss.item() * labels.size(0)
            pred = pred.argmax(1)
            correct += (pred == labels).sum().item()
            total += labels.numel()
    return (lossSum/total), (correct/total)

trainLosses, valLosses = [], []
bestValAcc, bestState = 0.0, None

#training loop
for epoch in range(1, epochs + 1):
    model.train()
    totalLoss, totalSamp = 0.0, 0
    for data, labels in trainDL:
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        pred = model(data)
        loss = criterion(pred, labels)
        loss.backward()
        optimizer.step()
        totalLoss += loss.item() * labels.size(0)
        totalSamp += labels.size(0)

    trainLoss = totalLoss / totalSamp
    valLoss, valAcc = eval(valDL)
    trainLosses.append(trainLoss)
    valLosses.append(valLoss)

    print(f"Epoch {epoch:02d} | trainLoss={trainLoss:.4f} | valLoss={valLoss:.4f} | valAcc={valAcc:.4f}")

    if valAcc > bestValAcc:
        bestValAcc = valAcc
        bestState = model.state_dict().copy()

if bestState is not None:
    model.load_state_dict(bestState)

#test
testLoss, testAcc = eval(testDL)
print(f"\nFinal TEST: loss={testLoss:.4f} | acc={testAcc:.4f}")

#save checkpoint
torch.save({
    "model": model.state_dict(),
    "scalerMean": scaler.mean_,
    "scalerScale": scaler.scale_,
    "labelToId": labelToId,
    "classes": classes,
    "inDim": inDim
}, "mlpGestures.pt")
print("Saved model to mlpGestures.pt")

#plot train/val loss
plt.figure()
plt.plot(range(1, epochs+1), trainLosses, label="trainLoss")
plt.plot(range(1, epochs+1), valLosses,   label="valLoss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train/Val Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("lossCurve.png")
print("Saved loss curve to lossCurve.png")