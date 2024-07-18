import matplotlib.pyplot as plt
import numpy as np
from backprop_network import *
from backprop_data import *

# Loading Data
np.random.seed(0)  # For reproducibility
n_train = 50000
n_test = 10000
x_train, y_train, x_test, y_test = load_as_matrix_with_labels(n_train, n_test)





# Training configuration
epochs = 20
batch_size = 100
learning_rate = 0.1

# Network configuration
layer_dims = [784, 1000, 500, 250, 100, 50, 10]
net = Network(layer_dims)
parameters, epoch_train_cost, epoch_test_cost, epoch_train_acc, epoch_test_acc = net.train(x_train, y_train, epochs, batch_size, learning_rate, x_test=x_test, y_test=y_test)

plt.figure(1)
plt.plot(np.linspace(1,epochs,epochs), epoch_test_acc, label="Test accuracy",color="blue")

plt.title("Test accuracy across epochs")
plt.xlabel("Epoch")
plt.legend()

print("Final test accuracy is:", epoch_test_acc[len(epoch_test_acc)-1])
plt.show()