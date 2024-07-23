import matplotlib.pyplot as plt
import numpy as np
from backprop_network import *
from backprop_data import *

# Loading Data
np.random.seed(0)  # For reproducibility



def main():
    '''
    After running a function, you should use plt.show() to see the corresponding graphs
    '''

    inp = input("Which part would you like to run? ")
    if(inp=='b'):
        partb()
    elif(inp=='c'):
        partc()
    elif(inp=='d'):
        partd()
    elif(inp=='e'):
        parte()

    plt.show()



def partb():
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


def partc():
    n_train = 50000
    n_test = 10000
    x_train, y_train, x_test, y_test = load_as_matrix_with_labels(n_train, n_test)

    # Training configuration
    epochs = 30
    batch_size = 100
    learning_rate = 0.1
    # Network configuration
    layer_dims = [784, 40, 10]
    net = Network(layer_dims)
    parameters, epoch_train_cost, epoch_test_cost, epoch_train_acc, epoch_test_acc = net.train(x_train, y_train, epochs, batch_size, learning_rate, x_test=x_test, y_test=y_test)

    plt.figure(1)
    plt.plot(np.linspace(1,30,30), epoch_test_acc, label="Test accuracy",color="blue")

    plt.title("Test accuracy across epochs")
    plt.xlabel("Epoch")
    plt.legend()

    print("Final test accuracy is:", epoch_test_acc[len(epoch_test_acc)-1])



def partd():
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




def parte():
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



if __name__ == '__main__':
    main()