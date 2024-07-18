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
epochs = 30
batch_size = 100
learning_rate = 0.1

# Network configuration
layer_dims = [784,10]
net = Network(layer_dims)
parameters, epoch_train_cost, epoch_test_cost, epoch_train_acc, epoch_test_acc = net.train(x_train, y_train, epochs, batch_size, learning_rate, x_test=x_test, y_test=y_test)

W = parameters['W1']

plt.figure(1)
plt.plot(np.linspace(1,30,30), epoch_train_acc, label="Training accuracy",color="red")
plt.plot(np.linspace(1,30,30), epoch_test_acc, label="Test accuracy",color="blue")

plt.title("Train and test accuracy across epochs")
plt.xlabel("Epoch")
plt.legend()


figure, axis = plt.subplots(2, 5) 
for i in range(2):
    for j in range(5):
        axis[i,j].imshow(np.reshape(W[i*5 + j], (28,28)), interpolation="nearest")
        axis[i,j].set_title("Row "+str(i*5 + j))
        axis[i,j].axis("off")

plt.suptitle("Rows of W as images")


plt.show()