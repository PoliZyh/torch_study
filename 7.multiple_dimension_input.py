import torch
import numpy as np

xy = np.loadtxt('../dataset/diabetes.csv', delimiter=',', dtype=np.float32)
# the first param is all lines, the second param is [begin, end) column
x_data = torch.from_numpy(xy[:, :-1])
# the first param is all lines, the second param is the end column, the [] can
# make sure that the column will be the matrix
y_data = torch.from_numpy(xy[:, [-1]])
# x_data = torch.tensor(xy[:, :-1], dtype=torch.float32)
# y_data = torch.tensor(xy[:, [-1]], dtype=torch.float32)


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

model = Model()

criterion = torch.nn.BCELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for epoch in range(100):
    # Forward
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print("epoch =", epoch, "loss =", loss.item())

    # Backward
    optimizer.zero_grad()
    loss.backward()

    # Update
    optimizer.step()

