# Notes: ABC Classification Neural Network

This notebook builds a very small neural network from scratch. It teaches the
network to recognize three simple pixel patterns: **A**, **B**, and **C**.
It uses only NumPy for calculations and Matplotlib for pictures and graphs.

## 1. Imports

```python
import numpy as np
import matplotlib.pyplot as plt
```

- **NumPy** works with arrays and performs the neural-network calculations.
- **Matplotlib** displays the letter patterns and training graphs.

## 2. Creating the dataset

Each letter is drawn on a grid with 5 rows and 6 columns. This gives
`5 * 6 = 30` pixels.

- `1` means that a pixel is on.
- `0` means that a pixel is off.

The notebook stores the three drawings in the lists `a`, `b`, and `c`.
Each list contains 30 numbers.

The target labels are:

```python
Y = [
    [1, 0, 0],  # A
    [0, 1, 0],  # B
    [0, 0, 1]   # C
]
```

These are called **one-hot labels**. Only one position is `1`, showing the
correct class.

## 3. Reshaping and displaying the letters

The flat lists are changed into 5-by-6 arrays:

```python
a_arr = np.array(a).reshape((5, 6))
b_arr = np.array(b).reshape((5, 6))
c_arr = np.array(c).reshape((5, 6))
```

`reshape((5, 6))` does not change the values. It only changes how they are
arranged so that they can be viewed as a grid.

The notebook displays each array with:

```python
plt.imshow(a_arr, cmap="gray")
plt.show()
```

`imshow` creates an image from the numbers. The gray color map makes the
patterns easy to see.

## 4. Preparing inputs and labels

The three letters are flattened again and stored in `x`:

```python
x = [
    np.array(a).reshape(1, 30),
    np.array(b).reshape(1, 30),
    np.array(c).reshape(1, 30)
]
```

Each input has shape `(1, 30)`:

- `1` is the number of examples being handled at that moment.
- `30` is the number of input pixels.

The neural network receives one letter at a time. The matching label is taken
from `Y` using the same position. For example, `x[0]` is A and `Y[0]` is
`[1, 0, 0]`.

### Small naming issue in the notebook

The labels are created as uppercase `Y`, but the display cell contains:

```python
y = np.array(y)
```

The lowercase `y` is not created in the shown code. It should normally be:

```python
Y = np.array(Y)
print(Y)
```

The training cell correctly uses uppercase `Y`.

## 5. Network structure

The network has three layers:

```text
Input layer:  30 neurons
Hidden layer: 5 neurons
Output layer: 3 neurons
```

The connections are fully connected. This means every neuron in one layer
connects to every neuron in the next layer.

- The 30 input neurons represent the 30 pixels.
- The 5 hidden neurons learn useful combinations of pixels.
- The 3 output neurons represent A, B, and C.

The output should look roughly like one of these:

```text
A -> [high, low,  low]
B -> [low,  high, low]
C -> [low,  low,  high]
```

## 6. The sigmoid activation function

```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
```

The sigmoid function changes any number into a value between 0 and 1. This is
useful because the output can be treated like a confidence value.

The derivative is:

```python
def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)
```

The derivative tells backpropagation how much a small change in an input
would change the sigmoid output.

## 7. Feedforward pass

The `feedforward` function sends an input through the network:

```python
z1 = np.dot(x, w1)
a1 = sigmoid(z1)
z2 = np.dot(a1, w2)
a2 = sigmoid(z2)
```

Step by step:

1. Multiply the input by the first weight matrix.
2. Apply sigmoid to get hidden-layer values (`a1`).
3. Multiply the hidden values by the second weight matrix.
4. Apply sigmoid again to get the final output (`a2`).

The weight shapes are:

```text
w1: (30, 5)   input to hidden layer
w2: (5, 3)    hidden layer to output layer
```

The dot product is valid because the inside dimensions match:

```text
(1, 30) dot (30, 5) -> (1, 5)
(1, 5)  dot (5, 3)  -> (1, 3)
```

## 8. Loss function

```python
def loss_function(y_pred, y_true):
    squared_error = np.square(y_pred - y_true)
    return np.mean(squared_error)
```

This uses **mean squared error (MSE)**:

1. Subtract the true label from the prediction.
2. Square the differences.
3. Find their average.

A smaller loss means that the prediction is closer to the correct label.

## 9. Creating weights

```python
def generate_weights(no_of_input_connections, no_of_output_connections):
    return np.random.randn(
        no_of_input_connections,
        no_of_output_connections
    )
```

The weights are the values the network learns. They start as random numbers so
the network does not begin with a fixed answer.

The notebook creates:

```python
w1 = generate_weights(30, 5)
w2 = generate_weights(5, 3)
```

Because the weights are random, a new run may start with different values and
may produce slightly different training results.

## 10. Backpropagation

Backpropagation updates the weights so that future predictions have less error.
The function first repeats the feedforward calculations:

```python
z1 = np.dot(x, w1)
a1 = sigmoid(z1)
z2 = np.dot(a1, w2)
a2 = sigmoid(z2)
```

It then calculates the output error:

```python
d2 = (a2 - actual) * sigmoid_derivative(z2)
```

The error is sent back to the hidden layer:

```python
d1 = np.dot(d2, w2.T) * sigmoid_derivative(z1)
```

The gradients show how each weight contributed to the error:

```python
w1_gradient = np.dot(x.T, d1)
w2_gradient = np.dot(a1.T, d2)
```

Finally, gradient descent changes the weights:

```python
w1 = w1 - learning_rate * w1_gradient
w2 = w2 - learning_rate * w2_gradient
```

The learning rate controls the size of each update. In the training call it is
`0.1`. A very large value can make learning unstable; a very small value can
make learning slow.

## 11. Training

The `train` function repeats learning for the requested number of epochs:

```python
loss, accuracy, w1, w2 = train(x, Y, w1, w2, 0.1, 1000)
```

An **epoch** is one complete pass over the three training examples. For every
example, the function:

1. Makes a prediction with `feedforward`.
2. Calculates the loss.
3. Updates both weight matrices with `backpropagation`.
4. Stores the loss and a value called accuracy.

The notebook runs 1,000 epochs. In the saved output, the last displayed loss
is about `0.00718`, and the displayed accuracy is about `99.992%`.

### Important note about the displayed accuracy

The code calculates:

```python
epoch_accuracy = (1 - (np.sum(epoch_loss) / np.size(x))) * 100
```

This is not a normal classification accuracy. It is a percentage made from
the MSE loss. True classification accuracy would compare the index of the
largest output value with the index of the largest target value, then count
the correct predictions.

Also, `epoch_loss` is replaced for each example, so the printed epoch loss is
the loss from the last example processed in that epoch. The `loss` list stores
values after each example, not one averaged value per epoch.

## 12. Training graphs

The notebook plots the saved values:

```python
plt.plot(accuracy)
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.show()

plt.plot(loss)
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.show()
```

The expected meaning is:

- The accuracy-like line should rise as learning improves.
- The loss line should fall as predictions get closer to the labels.

The graph has one stored value per training example, so it contains up to
`1000 * 3 = 3000` points.

## 13. Overall flow

```text
Create A, B, and C pixel patterns
        |
Convert lists to NumPy arrays
        |
Display the patterns
        |
Create one-hot labels
        |
Create random weights
        |
Feed input forward through 30 -> 5 -> 3 neurons
        |
Measure prediction error with MSE
        |
Use backpropagation to update weights
        |
Repeat for 1,000 epochs
        |
Plot the saved training values
```

## 14. Main ideas to remember

- A neural network uses numbers to represent data.
- The 30 pixels are the input features.
- Weights control how strongly one neuron affects another.
- The sigmoid function keeps activations between 0 and 1.
- Feedforward makes a prediction.
- Loss measures how wrong the prediction is.
- Backpropagation calculates useful weight changes.
- Gradient descent applies those changes.
- Repeating the process helps the network learn.
- This example is only a teaching example: it has three samples and no separate
  test set, so its training result should not be treated as real-world accuracy.
