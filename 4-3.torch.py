import torch

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

# If autograd mechanics are required, the element variable requires_grad of
# Tensor has to be set to True.
w = torch.Tensor([1.0])
w.requires_grad = True


def forward(x):
    return x * w

def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2

print("predict (before training)", 4, forward(4).item())

for epoch in range(100):
    for x, y in zip(x_data, y_data):
        l = loss(x, y)
        l.backward() # this code will storage the d(loss)/d(w) to the w.grad
        print("\tgrad:", x, y, w.grad.item())
        # to avoid update the Autograd(计算图), we should use .data to visit the value
        w.data = w.data - 0.01 * w.grad.data
        w.grad.data.zero_()
    print("progress:", epoch, l.item())

print("predict (after training)", 4, forward(4).item())

