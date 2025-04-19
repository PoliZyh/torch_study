import numpy as np
import matplotlib.pyplot as plt

# data-set
x_data = [1, 2, 3]
y_data = [2, 4, 6]
# the amount of samples
N = 3

w_list = []
MSE_list = []

def linear_model(x, w):
    y_pred = x * w
    return y_pred

def loss(y, y_pred):
    return (y - y_pred) * (y - y_pred)

def main():
    # For each w, calculate the loss of all samples and calculate the MSE
    for w in np.arange(0.0, 4.1, 0.1):
        loss_sum = 0
        for x_val, y_val in zip(x_data, y_data):
            y_pred = linear_model(x_val, w)
            loss_i = loss(y_val, y_pred)
            loss_sum += loss_i
        w_list.append(w)
        MSE_list.append(loss_sum / N)

    # plot the MSE curve
    plt.plot(w_list, MSE_list)
    plt.title("MSE Curve")
    plt.ylabel("MSE")
    plt.xlabel("w")
    plt.show()



if __name__ == '__main__':
    main()






