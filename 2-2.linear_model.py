import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# data-set
x_data = [1, 2, 3]
y_data = [2, 4, 6]
# the amount of samples
N = 3




def forward(x, w, b):
    return x * w + b

def loss(y, y_pred):
    return (y - y_pred) * (y - y_pred)

def main():
    W = np.arange(0.0, 4.0, 0.01)
    B = np.arange(-2.0, 2.0, 0.01)
    # the shape of w and b is metrix, len(B) * len(W)
    w, b = np.meshgrid(W, B)
    mse_matrix = np.zeros_like(w)
    for i in range(len(B)):
        for j in range(len(W)):
            loss_sum = 0
            for x_val, y_val in zip(x_data, y_data):
                y_pred = forward(x_val, w[i, j], b[i, j])
                loss_k = loss(y_val, y_pred)
                loss_sum += loss_k
            mse_matrix[i, j] = loss_sum / N

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(w, b, mse_matrix)
    ax.set_xlabel('w')
    ax.set_ylabel('b')
    ax.set_zlabel('MSE Loss')
    ax.set_title('3D Visualization of MSE Loss Function')
    plt.show()



if __name__ == '__main__':
    main()