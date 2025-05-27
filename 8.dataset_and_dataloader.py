import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np

class DiabetesDataset(Dataset):
    def __init__(self, filepath):
        xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        # shape[0] is the number of rows
        # shape[1] is the number of cols
        self.len = xy.shape[0]
        # self.x_data = torch.from_numpy(xy[:, :-1])
        # self.y_data = torch.from_numpy(xy[:, [-1]])
        self.x_data = torch.tensor(xy[:, :-1], dtype=torch.float32)
        self.y_data = torch.tensor(xy[:, [-1]], dtype=torch.float32)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        y_pred = self.sigmoid(self.linear3(x))
        return y_pred



dataset = DiabetesDataset('../dataset/diabetes.csv')
train_loader = DataLoader(dataset=dataset,
                          batch_size=32,
                          shuffle=True,
                          num_workers=2)
model = Model()
criterion = torch.nn.BCELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

if __name__ == "__main__":
    for epoch in range(1000):
        for i, data in enumerate(train_loader, 0):
            # Prepare data
            inputs, labels = data
            # Forward
            y_pred = model(inputs)
            loss = criterion(y_pred, labels)
            print("epoch =", epoch, "i =", i, "loss =", loss.item())
            # Backward
            optimizer.zero_grad()
            loss.backward()
            # Update
            optimizer.step()