import numpy as np
import matplotlib.pyplot as plt


# data-set
x_data = [1, 2, 3]
y_data = [2, 4, 6]



def forward(x, w):
    y_pred = w * x
    return y_pred

def loss(x, y, w):
    y_pred = forward(x, w)
    return  (y_pred - y) ** 2

def gradient(x, y, w):
    grad = 2 * x * (x * w - y)
    return grad

def main():
    w = 1
    alpha = 0.01
    for epoch in range(100):
        for x, y in zip(x_data, y_data):
            grad_val = gradient(x, y, w)
            w -= alpha * grad_val
            loss_val = loss(x, y, w)
        print('epoch=', epoch, 'w=', w, 'loss=', loss_val)


if __name__ == '__main__':
    main()

