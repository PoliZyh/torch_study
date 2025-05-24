import torch

# Step 1
x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[2.0], [4.0], [6.0]])

# Step 2
class LinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

model = LinearModel()

# Step 3
# set the way of calculating loss
criterion = torch.nn.MSELoss(reduction='sum')
# find all params, and set the learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Step 4
for epoch in range(100):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print("epoch =", epoch, "loss = ", loss)
    # Before backward, remember set the grad to ZERO!!!
    optimizer.zero_grad()
    # Calculate the grad
    loss.backward()
    # update params
    optimizer.step()

# y_pred --> loss --> backward --> update

# Step 5
print("w =", model.linear.weight.item())
print("b =", model.linear.bias.item())

x_test = torch.Tensor([4.0])
y_test = model(x_test)
print("y_pred =", y_test.item())