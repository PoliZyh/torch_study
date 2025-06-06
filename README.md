# Pytorch_study



## 1. 基本知识

### 1.1 基本流程

- 收集数据集DataSet
- 选择模型Model
- 训练Training
- 推理Inferring



### 1.2 数据集划分

| 训练集（x, y） | 测试集(x) |
| :------------: | :-------: |

| 训练集(x,y) | 开发集/验证集(x,y) | 测试集(x) |
| :---------: | :----------------: | :-------: |



### 1.3 损失函数/误差函数/残差 (Traning Loss)


$$
loss = (\hat{y} - y)^2
$$

$$
\text{MSE} = \text{cost} = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)^2
$$

Where, loss is one sample error; MSE is mean squared error, n is the amount of samples.



## 2. Linear Model

$$
\hat{y} = x \cdot \omega + b
$$

Where, the machine starts with a random guess.
$$
\omega = Random \  \ Value
$$

$$
\hat{y} = f(x)
$$



### 2-1. Q: Suppose that students would get Y points in final exam, if they spent X hovers in paper PyTorch Tutorial.

| X(hours) | Y(points) |
| :------: | :-------: |
|    1     |     2     |
|    2     |     4     |
|    3     |     6     |
|    4     |    ???    |

- The question is what would be the grade if I study 4 hours?

**A: Enumerating Method! We can find that the best parameters are located within an interval. So, we can enumerate all possibilities and plot the loss function curve, with the lowest point being the best parameters.**

<img src="./imgs/2-1_1.png">

From the image, it can be concluded that when the value of w is 2.0, the loss function is minimized, therefore this point is the best parameter.



### 2-2 Q: Try to use the model in (3), and draw the cost graph.

**Tips:**

- You can read the material of how to draw 3d graph.
- Function **np.meshgrid()** is very popular for drawing 3d graph, read the docs and utilize vectorization calculation


<div align="center">
<img src="./imgs/2-2_2.png">
</div>


## 3. Gradient Descent

$$
\text{Gradient}=\frac{\partial \cos t}{\partial w}
$$

Where,  cost is y-axis and w is x-axis.

So, how to **update** ?
$$
w = w - \alpha \times \frac{\partial \cos t}{\partial w}
$$
Where, alpha is the learning rate and it should be set to a smaller value.
$$
\frac{\partial \text{cost}(\omega)}{\partial \omega} = \frac{\partial}{\partial \omega} \frac{1}{N} \sum_{n=1}^{N} (x_n \cdot \omega - y_n)^2

= \frac{1}{N} \sum_{n=1}^{N} \frac{\partial}{\partial \omega} (x_n \cdot \omega - y_n)^2

= \frac{1}{N} \sum_{n=1}^{N} 2 \cdot (x_n \cdot \omega - y_n) \frac{\partial (x_n \cdot \omega - y_n)}{\partial \omega}

= \frac{1}{N} \sum_{n=1}^{N} 2 \cdot x_n \cdot (x_n \cdot \omega - y_n)
$$

### 3-1. Q: Try to use the gradient descent to find the MSE.
<div align="center"><img src="./imgs/3-1_1.png"></div>

### 3-2. Q: Try to use the stochastic gradient descent to find the MSE.

What is **Stochastic Gradient Descent**?
$$
\text{Stochastic Gradient}=\frac{\partial loss}{\partial w}
$$
Where, loss is the loss value of a random sample.

So, how to update?
$$
w = w - \alpha \times \frac{\partial loss}{\partial w}
$$

$$
\frac{\partial  loss_n}{\partial \omega} = 2 \cdot x_n \cdot (x_n \cdot \omega - y_n)
$$

By using stochastic gradient descent, we can avoid getting stuck at saddle points.

### 3-3. What is Batch and Mini-Batch?

Group the dataset and calculate the gradient of each group for updating at each time step.
<div align="center">
<img src="./imgs/3-3_1.png">
</div>


## 4. Back Propagation

A two layer neural network is
$$
\hat{y} = W_2(W_1 \cdot X + b_1) + b_2
$$
Where, W is the weight matrix and b is the bias.

<div align="center">
<img src="./imgs/4_1.png">
</div>
### 4-1. Q: Compute the gradient with Computational Graph.

<div align="center">
<img src="./imgs/4-1_1.png">
</div>

A:
<div align="center">
<img src="./imgs/4-1_2.jpg">
</div>


### 4-2. Q: Compute gradient of Affine model.
<div align="center">
<img src="./imgs/4-2_1.png">
</div>
A:
<div align="center">
<img src="./imgs/4-2_2.jpg">
</div>

### 4-3. What is Tensor?

In PyTorch, Tensor is the important component in constructing dynamic computational graph.

It contains data and grad, which storage the value of node and gradient w.r.t loss respectively.

- **用 `.data`**：
  - 需要修改张量的值（如更新模型权重 `weight.data -= lr * weight.grad`）。
  - 需要访问张量的数据但不想影响梯度计算。
- **用 `.item()`**：
  - 需要将单元素张量转换为 Python 数值（如打印 loss、计算指标）。

### 4-4. Q: Compute gradients using PyTorch.
<div align="center">
<img src="./imgs/4-4_1.png">
</div>
## 5. Linear Regression with PyTorch

## 6. Logistic Regression

Although it is called "regression", it is used for classification tasks

In classification, the output of model is the **probability** of input belongs to the exact class.

How to map? R --> [0, 1]? **Use Logistic Function!**
$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$
<div align="center">
<img src="./imgs/5_1.png">
</div>

<div align="center">
<img src="./imgs/5_2.png">
</div>

<div align="center">
<img src="./imgs/5_3.png">
</div>

The Loss Function is called BCE_Loss.

<div align="center">
<img src="./imgs/5_4.png">
</div>
Test the model...

| x(hours) | y(pass/fail) |
| -------- | ------------ |
| 1        | 0(fail)      |
| 2        | 0(fail)      |
| 3        | 1(pass)      |
| 4        | ?            |
| ...      | ?...         |



<div align="center">
<img src="./imgs/5_5.png">
</div>
## 7. Multiple Dimension Input

DataSet:
<div align="center">
<img src="./imgs/6_1.png">
</div>

<div align="center">
<img src="./imgs/6_2.png">
</div>

notice! The first param of Linear is the dimension of input and the second param of Linear is the dimension of the output.

<div align="center">
<img src="./imgs/6_3.png">
</div>

The new model.

<div align="center">
<img src="./imgs/6_4.png">
</div>

Change the activate function.

<div align="center">
<img src="./imgs/6_5.png">
</div>

## 8. Dataset and DataLoader

### 8-1. Build a classifier using the DataLoader and Dataset 《titanic》.

<div align="center">
<img src="./imgs/8-1_1.png">
</div>
## 9. Softmax Classifier

Output a Distribution of prediction with Softmax.

<div align="center">
<img src="./imgs/9_1.png">
</div>

The Softmax function:
$$
 P(y = i) = \frac{e^{z_i}}{\sum_{j=0}^{K-1} e^{z_j}}, \quad i \in \{0, \dots, K-1\} 
$$

<div align="center">
<img src="./imgs/9_2.png">
</div>

Loss.
<div align="center">
<img src="./imgs/9_3.png">
</div>
CrossEntropyLoss.

<div align="center">
<img src="./imgs/9_4.png">
</div>

$$
\text{CrossEntropyLoss} \Longleftrightarrow \text{LogSoftmax} + \text{NLLLoss}
$$

### 9-1. Implementation of classifier to MNIST dataset.
<div align="center">
<img src="./imgs/9-1_1.png">
</div>
Use transform to convert the PIL Image to Tensor. W * H * C --> C * W * H.

The params of transform are **mean** and **std** respectively.

<div align="center">
<img src="./imgs/9-1_2.png">
</div>
The model is.

<div align="center">
<img src="./imgs/9-1_3.png">
</div>

<div align="center">
<img src="./imgs/9-1_4.png">
</div>
### 9-2. Try to implement a classifier for Otto.

## 10. Basic CNN
<div align="center">
<img src="./imgs/10_1.png">
</div>
Single Input Channel.
<div align="center">
<img src="./imgs/10_2.png">
</div>

3 Input Channels.
<div align="center">
<img src="./imgs/10_3.png">
</div>

What is the Convolution?
<div align="center">
<img src="./imgs/10_4.png">
</div>

<div align="center">
<img src="./imgs/10_5.png">
</div>

N input Channels and M output Channels.

<div align="center">
<img src="./imgs/10_6.png">
</div>


<div align="center">
<img src="./imgs/10_7.png">
</div>

<div align="center">
<img src="./imgs/10_8.png">
</div>

torch.Size([1, 5, 100, 100]).    [batch, channel, height, width]
torch.Size([1, 10, 98, 98]).  [batch, channel, height, width]
torch.Size([10, 5, 3, 3]).   [output_channel, input_channel, kernel_height, kernel_width]

### 10-1. Padding
<div align="center">
<img src="./imgs/10-1_1.png">
</div>
### 10-2. Stride
<div align="center">
<img src="./imgs/10-2_1.png">
</div>
### 10-3. Max Pooling Layer
<div align="center">
<img src="./imgs/10-3_1.png">
</div>
### 10-4. A Simple Convolutional Neural Network
<div align="center">
<img src="./imgs/10-4_1.png">
</div>

<div align="center">
<img src="./imgs/10-4_2.png">
</div>

<div align="center">
<img src="./imgs/10-4_3.png">
</div>

[10,   300] loss: 0.038
[10,   600] loss: 0.036
[10,   900] loss: 0.039
Accuracy on test: 98.720 %

## 11. Advanced CNN

### 11-1. GoogLe Net

Inception Module
<div align="center">
<img src="./imgs/11-1_1.png">
</div>
Concatenate, make sure that **only C** can be different. [B, C, W, H]

What is 1✖️1 convolution?
<div align="center">
<img src="./imgs/11-1_2.png">
</div>

Why is  1✖️1 convolution?
<div align="center">
<img src="./imgs/11-1_3.png">
</div>

<div align="center">
<img src="./imgs/11-1_4.png">
</div>

<div align="center">
<img src="./imgs/11-1_5.png">
</div>

<div align="center">
<img src="./imgs/11-1_6.png">
</div>

<div align="center">
<img src="./imgs/11-1_7.png">
</div>

How to get 1408? Delete the three rows which is marked and get the error informations.



### 11-2. Residual Net

<div align="center">
<img src="./imgs/11-2_1.png">
</div>
**prevent gradient vanishing**
Residual Block.

<div align="center">
<img src="./imgs/11-2_2.png">
</div>

<div align="center">
<img src="./imgs/11-2_3.png">
</div>