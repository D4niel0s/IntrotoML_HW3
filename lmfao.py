import matplotlib.pyplot as plt
import numpy as np
from backprop_network import *
from backprop_data import *


x_train, y_train, x_test, y_test = load_as_matrix_with_labels(10000, 5000)


# Training configuration
epochs = 30
batch_size = 10
learning_rate = 0.1

# Network configuration
layer_dims = [784, 40, 10]
net = Network(layer_dims)
parameters, epoch_train_cost, epoch_test_cost, epoch_train_acc, epoch_test_acc = net.train(x_train, y_train, epochs, batch_size, learning_rate, x_test=x_test, y_test=y_test)

ep = np.linspace(1,30,30)
plt.plot(ep,epoch_train_cost, label="Training loss", color="red")
plt.plot(ep,epoch_train_acc, label="Training accuracy", color="blue")
plt.plot(ep,epoch_test_acc, label="Test accuracy", color="green")
plt.xlabel("Epoch")
plt.legend()
plt.show()