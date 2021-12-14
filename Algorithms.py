import numpy as np
from numpy.core.fromnumeric import take
import scipy


def sigmoid(x):
    return scipy.special.expit(x)


def Cost_Function(nnParam, input_layer_size, hidden_layer_size, num_labels, X, y, lambdaa, key_change_Y, return_grad=False):
    Theta1 = np.reshape(nnParam[:hidden_layer_size * (input_layer_size + 1)],
                        (hidden_layer_size, input_layer_size + 1))
    Theta2 = np.reshape(
        nnParam[hidden_layer_size * (input_layer_size + 1):], (num_labels, hidden_layer_size + 1))

    m = len(X)

    J = 0
    Theta1_grad = np.zeros(Theta1.shape)
    Theta2_grad = np.zeros(Theta2.shape)

    a1 = np.concatenate((np.ones((m, 1)), X), axis=1)

    z2 = a1.dot(Theta1.T)
    a2 = np.concatenate((np.ones((m, 1)), sigmoid(z2)), axis=1)

    z3 = a2.dot(Theta2.T)
    a3 = sigmoid(z3)

    y_matrix = np.zeros((m, num_labels))

    for i in range(m):
        y_matrix[i, y[i]] = 1

    if key_change_Y == True:
        b = y_matrix[:, 0]
        y_matrix = y_matrix[:, 1:]
        y_matrix = np.column_stack((y_matrix, b))

    J = (1 / m) * np.sum(np.sum(-y_matrix * np.log(a3) -
                                (1 - y_matrix) * np.log(1 - a3), axis=1))
    J += (lambdaa / (2 * m)) * \
        (np.sum(np.sum(Theta1[:, 1:] ** 2, axis=1)) +
         np.sum(np.sum(Theta2[:, 1:] ** 2, axis=1)))
    
    if return_grad == True:
        delta3 = a3 - y_matrix
        delta2 = delta3.dot(Theta2[:, 1:]) * sigmoid(z2) * (1 - sigmoid(z2))

        Reg2 = lambdaa * \
            np.concatenate((np.zeros((Theta2.shape[0], 1)), Theta2[:, 1:]), axis=1)
        Reg1 = lambdaa * \
            np.concatenate((np.zeros((Theta1.shape[0], 1)), Theta1[:, 1:]), axis=1)

        Theta2_grad = (1 / m) * ((a2.T.dot(delta3)).T + Reg2)
        Theta1_grad = (1 / m) * ((a1.T.dot(delta2)).T + Reg1)

        grad = np.concatenate((Theta1_grad.flatten(), Theta2_grad.flatten()))

        delta3 = a3 - y_matrix
        delta2 = delta3.dot(Theta2[:, 1:]) * sigmoid(z2) * (1 - sigmoid(z2))

        Reg2 = lambdaa * \
            np.concatenate(
                (np.zeros((Theta2.shape[0], 1)), Theta2[:, 1:]), axis=1)
        Reg1 = lambdaa * \
            np.concatenate(
                (np.zeros((Theta1.shape[0], 1)), Theta1[:, 1:]), axis=1)

        Theta2_grad = (1 / m) * ((a2.T.dot(delta3)).T + Reg2)
        Theta1_grad = (1 / m) * ((a1.T.dot(delta2)).T + Reg1)

        grad = np.concatenate((Theta1_grad.flatten(), Theta2_grad.flatten()))
        return [J, grad]
    else:
        return J


def Theta_Grad(nnParam, input_layer_size, hidden_layer_size, num_labels, X, y, lambdaa, temp1, temp2):
    Theta1 = np.reshape(nnParam[:hidden_layer_size * (input_layer_size + 1)],
                        (hidden_layer_size, input_layer_size + 1))
    Theta2 = np.reshape(
        nnParam[hidden_layer_size * (input_layer_size + 1):], (num_labels, hidden_layer_size + 1))

    m = len(X)

    J = 0
    Theta1_grad = np.zeros(Theta1.shape)
    Theta2_grad = np.zeros(Theta2.shape)

    a1 = np.concatenate((np.ones((m, 1)), X), axis=1)

    z2 = a1.dot(Theta1.T)
    a2 = np.concatenate((np.ones((m, 1)), sigmoid(z2)), axis=1)

    z3 = a2.dot(Theta2.T)
    a3 = sigmoid(z3)

    y_matrix = np.zeros((m, num_labels))

    for i in range(m):
        y_matrix[i, y[i]] = 1

    delta3 = a3 - y_matrix
    delta2 = delta3.dot(Theta2[:, 1:]) * sigmoid(z2) * (1 - sigmoid(z2))

    Reg2 = lambdaa * \
        np.concatenate((np.zeros((Theta2.shape[0], 1)), Theta2[:, 1:]), axis=1)
    Reg1 = lambdaa * \
        np.concatenate((np.zeros((Theta1.shape[0], 1)), Theta1[:, 1:]), axis=1)

    Theta2_grad = (1 / m) * ((a2.T.dot(delta3)).T + Reg2)
    Theta1_grad = (1 / m) * ((a1.T.dot(delta2)).T + Reg1)

    grad = np.concatenate((Theta1_grad.flatten(), Theta2_grad.flatten()))

    return grad


def Predict(Theta1, Theta2, p):

    a1 = np.concatenate(([1], p))

    z2 = a1.dot(Theta1.T)

    a2 = np.concatenate((np.ones((1)), sigmoid(z2)))

    z3 = a2.dot(Theta2.T)
    a3 = sigmoid(z3)

    return np.argmax(a3)


def Accuracy(X, Theta1, Theta2, y):
    m = len(X)
    correct = 0
    wrong = 0
    for i in range(m):
        if Predict(Theta1, Theta2, X[i]) == y[i]:
            correct += 1
        else:
            wrong += 1
    return [correct, wrong]