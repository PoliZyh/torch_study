import numpy as np
import matplotlib.pyplot as plt


# data-set
x_data = [1, 2, 3]
y_data = [2, 4, 6]

cost_list = []
epoch_list = []

def forward(x, w):
    y_pred = w * x
    return y_pred

def cost(xs, ys, w):
    loss_sum = 0
    for x, y in zip(xs, ys):
        y_pred = forward(x, w)
        loss_sum += (y_pred - y) ** 2
    return loss_sum / len(xs)

def gradient(xs, ys, w):
    grad = 0
    for x, y in zip(xs, ys):
        grad += 2 * x * (x * w - y)
    return grad / len(xs)

def main():
    w = 1
    alpha = 0.01
    for epoch in range(100):
        cost_val = cost(x_data, y_data, w)
        grad_val = gradient(x_data, y_data, w)
        w -= alpha * grad_val
        epoch_list.append(epoch)
        cost_list.append(cost_val)
    plt.plot(epoch_list, cost_list)
    plt.title('Cost in each epoch')
    plt.xlabel('epoch')
    plt.ylabel('cost')
    plt.show()

if __name__ == '__main__':
    main()
