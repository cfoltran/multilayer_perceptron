import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import copy
# Get the dataset
df = pd.read_csv('data.csv').values
# Remove patient ID column
df = np.delete(df, 0, 1)

X = df[:, 1:].T.astype(float)
print(X.shape)
Y = df[:, 0].T.reshape(1, len(df))

# Binary representation of diagnosis 0 for benign 1 for malignant
Y = np.where(Y == 'M', 1, 0)
print(Y.shape)

# Data normalization (minmax)
vmin = np.min(X, axis=1, keepdims=True)
vmax = np.max(X, axis=1, keepdims=True)
X = (X - vmin) / (vmax - vmin)
# Constants defining the model
n_x = X.shape[0]
n_h = 7
n_y = 1
learning_rate = .0075
n_iterations = 3000
def initialiaze_parameters(n_x, n_h, n_y):
    """
    Argument:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer

    Returns:
    params -- python dictionary containing the parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
    """
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1= np.random.randn(n_h, 1) * 0.01
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.random.randn(n_y, 1) * 0.01

    return {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2
    }

def initialize_parameters_deep(layer_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network

    Returns -- python dic containing W1, b1...
    Wl -- weight matrix
    bl -- bias vector
    """

    parameters = {}
    
    for l in range(1, len(layer_dims)):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(layer_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

    return parameters;
def sigmoid(z):
    """
    compute de sigmoid of z

    Arguments:
    z -- A scalar of np array of any size

    Return:
    sigmoid(z)
    """
    A = 1 / (1 + np.exp(-z))
    cache = z

    return A, cache
# assert(np.allclose(sigmoid(np.array([0, 2])), np.array([0, 2]), np.array([0.5, 0.88079708]), atol=1e-7))

def relu(z):
    """
    compute de relu of z

    Arguments:
    z -- A scalar of np array of any size

    Return:
    relu(z)
    """
    A = np.maximum(0, z)
    cache = z

    return A, cache

assert(relu(np.array([2, 10, 20])) == np.array([2, 10, 20])).all()
def linear_forward(A, W, b):
    """
    Linear forward propagation

    Arguments:
    A -- activation from previous layer 
    W -- weight matrix, np.array().shape((size of current layer, size of previous layer))
    b -- bias vector, np.array().shape((size of current layer, 1))
    """
    Z = W.dot(A) + b
    cache = (A, W, b)

    return Z, cache

def linear_activation_forward(A_prev, W, b, activation):
    if (activation == 'sigmoid'):
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)

    elif (activation == 'relu'):
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    cache = (linear_cache, activation_cache)

    return A, cache

def multilayers_model_forward(X, parameters): 
    caches = []
    A = X
    L = len(parameters) // 2

    for l in range(1, L):
        A_prev = A 
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], "relu")
        caches.append(cache)
    
    AL, cache = linear_activation_forward(A, parameters['W' + str(l)], parameters['b' + str(l)], 'sigmoid')
    caches.append(cache)

    return AL, cache
def compute_cost(AL, Y):
    m = Y.shape[1]

    cost = -1 / m * np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL))

    return np.squeeze(cost);
def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = (1 / m) * dZ.dot(A_prev.T)
    db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ = np.array(dA, copy=True)
        dZ[activation_cache <= 0] = 0
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    elif (activation == 'sigmoid'):
        s = 1 / (1 + np.exp(-activation_cache))
        dZ = dA * s * (1 - s)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    

    return dA_prev, dW, db

def multilayer_model_backward(AL, Y, caches):
    """
    Initializing backpropagation
    """
    grads = {}
    L = len(caches)
    # Y = Y.reshape(AL.shape)

    # derivative of cost
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    curr_cache = caches[L - 1]
    grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, curr_cache, activation='sigmoid')

    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache, activation = "relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads
def update_parameters(params, grads, learning_rate):
    parameters = copy.deepcopy(params)
    L = len(parameters) // 2

    for l in range(L):
        parameters["W" + str(l)] = parameters["W" + str(l)] - learning_rate * grads["dW" + str(l)]
        parameters["b" + str(l)] = parameters["b" + str(l)] - learning_rate * grads["db" + str(l)]

    return parameters
def multilayer_model(X, Y, layers_dims, learning_rate = .0075, num_iterations = 3000, print_cost=False):
    """
    Multilayer neural networks

    Arguments:
    X -- input data, of shape (n_x, number of examples)
    Y -- label vector containing (1 if Malignant cancer, 0 either), of shape (1, number of examples)
    """
    costs = []
    parameters = initialize_parameters_deep(layers_dims)
    
    # navigate through each layers
    for i in range(1, 3): # 3️⃣ hardcoded 
        AL, caches = multilayers_model_forward(X, parameters)
        
        cost = compute_cost(AL, Y)

        print("cost: " + str(cost))
        
        gradients = multilayer_model_backward(AL, Y, caches)

        parameters = update_parameters(parameters, gradients, learning_rate)

    return parameters, cost
layers_dims = [30, 30, 30, 1]
parameters, cost = multilayer_model(X, Y, layers_dims)

print(cost)
print(parameters)
