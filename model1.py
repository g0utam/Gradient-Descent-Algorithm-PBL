import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Define the linear neuron function
def linear_neuron(x, w, b):
    return w * x + b


# Define the loss function
def compute_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


# Define the gradient descent function with tracking of weight and bias path
def gradient_descent_tracking(x, y, w_init, b_init, learning_rate, iterations):
    w = w_init
    b = b_init
    n = len(y)
    loss_history = []
    path_w, path_b = [], []

    for i in range(iterations):
        y_pred = linear_neuron(x, w, b)
        loss = compute_loss(y, y_pred)
        loss_history.append(loss)

        # Store the current weights and biases
        path_w.append(w)
        path_b.append(b)

        # Compute gradients
        dw = -(2 / n) * np.sum(x * (y - y_pred))
        db = -(2 / n) * np.sum(y - y_pred)

        # Update parameters
        w -= learning_rate * dw
        b -= learning_rate * db

    return w, b, loss_history, path_w, path_b

# Load data from CSV file
df = pd.read_csv('D:\COLLEGE\SEM 5\PBL\FLNN\sample_data.csv')  # Replace 'data.csv' with your actual file path
x = df['feature'].values  # Replace 'feature' with your actual column name
y = df['target'].values  # Replace 'target' with your actual column name

learning_rates = [0.001, 0.01, 0.1]
w_init = 0.0
b_init = 0.0
iterations = 100

# Plot the loss convergence for different learning rates
plt.figure(figsize=(12, 8))

for lr in learning_rates:
    w, b, loss_history, path_w, path_b = gradient_descent_tracking(x, y, w_init, b_init, lr, iterations)
    plt.plot(loss_history, label=f"Learning Rate: {lr}")
    final_loss = loss_history[-1]
    print(f"Final loss for learning rate {lr}: {final_loss}")

plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Gradient Descent Convergence')
plt.legend()
plt.show()

# Plot the path of weights and biases during gradient descent
plt.figure(figsize=(12, 8))

for lr in learning_rates:
    _, _, _, path_w, path_b = gradient_descent_tracking(x, y, w_init, b_init, lr, iterations)
    plt.plot(path_w, path_b, marker='o', label=f'LR: {lr}')

plt.xlabel('Weight (w)')
plt.ylabel('Bias (b)')
plt.title('Path of Weights and Biases During Gradient Descent')
plt.legend()
plt.show()