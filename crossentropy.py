import helper1
import numpy as np

# No dropout to start with, it shall be added later same with L2 regularization

M = 1  # Number of training samples
TRAIN_FILE_PATH = 'C:/Users/user/Downloads/MSINT CSV/mnist_train.csv'


labels, a0 = helper1.ReadCSVFile(TRAIN_FILE_PATH, M)


def First_Layer(X, W, b, M):
    Z = np.zeros((M, W.shape[1]))
    activations = np.zeros((M, W.shape[0]))

    for i in range(M):
        Z[i] = helper1.Z(W, X[i], b)
        if np.max(Z[i]) != 0:
            Z[i] = np.divide(Z[i], np.max(Z[i]))
        activations[i] = helper1.ReLu(Z[i])
    return activations, Z # (M, 784)

def Output_Layer(X, W, b, M):
    Z = np.zeros((M, W.shape[0]))
    activations = np.zeros((M, W.shape[0]))

    for i in range(M):
        Z[i] = helper1.Z(W, X[i], b)
        if np.max(Z[i]) != 0:
            Z[i] = np.divide(Z[i], np.max(Z[i]))
    for i in range(M):
        activations[i] = helper1.Softmax(Z[i],Z)
    return activations, Z # (M, 10)

def CrossEntropyCost():
    pass # do later

def ELL(output, labels, M):
    return np.multiply(np.subtract(output, 1), helper1.CorrectOutput(labels, M)) # (M, 10)

def EFL(W2, Z1, output, labels, M):
    return np.dot(np.dot(W2.T, ELL(output, labels, M).T).T, helper1.DerrivativeSoftmax(Z1))

def CostWithBias(layer, W2, Z1, output, M):
    if layer == 1:
        return np.mean(EFL(W2, Z1, output, labels, M))
    elif layer == 2:
        return np.mean(ELL(output, labels, M))
    return None
    
def CostWithWeight(layer, a0, a1, W2, Z1, output):
    if layer == 1:
        return np.dot(EFL(W2, Z1, output, labels, M).T, a0)
    if layer == 2:
        return np.dot(ELL(output, labels, M).T, a1)
    return None

def CostGradient(a0, a1, W2, Z1, output,M):
    W1t = CostWithWeight(1, a0, a1, W2, Z1, output).reshape(-1)
    W2t = CostWithWeight(2, a0, a1, W2, Z1, output).reshape(-1)
    B1t = CostWithBias( 1, W2, Z1, output, M)
    B2t = CostWithBias(2, W2, Z1, output, M)

    return np.r_[W1t, W2t, B1t, B2t]

def Theta():
    return np.r_[w1.reshape(-1), w2.reshape(-1), b1, b2]

def GradientDescent(a0, a1, W2, Z1, output, STEP_SIZE,M):
    theta = Theta()
    theta += -STEP_SIZE * RMSprop(a0, a1, W2, Z1, output,M)

    w1t = theta[:614656].reshape((784, 784))
    w2t = theta[614656:622496].reshape((10, 784))
    b1t = theta[-2]
    b2t = theta[-1]
    return w1t, w2t, b1t, b2t

def RMSprop(a0, a1, W2, Z1, output,M):
    global E
    g = CostGradient(a0, a1, W2, Z1, output,M)
    E = 0.9 * E + 0.1 * np.square(g)
    return g / (np.sqrt(E + 1e-8))

# Weight and bias initialziations
w1 = helper1.Xavier_initialization((784, 784))
w2 = helper1.Xavier_initialization((10, 784))
b1 = b2 = 1.0 

E = np.zeros(622498)
step_size = 0.1


for i in range(10):
    a1, z1 = First_Layer(a0, w1, b1, M)
    a2, z2 = Output_Layer(a1, w2, b2, M)
    w1, w2, b1, b2 = GradientDescent(a0, a1, w2, z1, a2, step_size,M)

    
    print(f"5 = {a2[0]}")
    print("-----------------------")

'''
a1, z1 = First_Layer(a0, w1, b1, M)
a2, z2 = Output_Layer(a1, w2, b2, M)
'''


# FUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUCKKKKKKKKKKKKKKK