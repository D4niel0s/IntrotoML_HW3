import matplotlib.pyplot as plt
import numpy as np
from backprop_network import *
from backprop_data import *


x_train, y_train, x_test, y_test = load_as_matrix_with_labels(10000, 5000)


# Training configuration
epochs = 30
batch_size = 10

# Network configuration
layer_dims = [784, 40, 10]


ep_range = np.linspace(1,30,30)

train_accs = [None]*5
train_losses = [None]*5
test_accs = [None]*5

cur_rate = 10
for i in range(5):
    net = Network(layer_dims)
    parameters, epoch_train_cost, epoch_test_cost, epoch_train_acc, epoch_test_acc = net.train(x_train, y_train, epochs=epochs, batch_size=batch_size, learning_rate=cur_rate, x_test=x_test, y_test=y_test)
    
    train_accs[i] = epoch_train_acc
    train_losses[i] = epoch_train_cost
    test_accs[i] = epoch_test_acc

    cur_rate /= 10


plt.figure(1)
plt.title("Training accuracy across learning rates")
cur_rate = 10
for i in range(5):
    plt.plot(ep_range, train_accs[i], label="LR="+str(cur_rate))
    cur_rate /= 10

plt.xlabel("Epochs")
plt.legend()
#-------------------------
plt.figure(2)
plt.title("Training loss across learning rates")
cur_rate = 10
for i in range(5):
    plt.plot(ep_range, train_losses[i], label="LR="+str(cur_rate))
    cur_rate /= 10

plt.xlabel("Epochs")
plt.legend()
#-------------------------
plt.figure(3)
plt.title("Test accuracy across learning rates")
cur_rate = 10
for i in range(5):
    plt.plot(ep_range, test_accs[i], label="LR="+str(cur_rate))
    cur_rate /= 10

plt.xlabel("Epochs")
plt.legend()

print("Test accuracy with LR=0.1 is: ", test_accs[2][len(test_accs[2])-1])

plt.show()

