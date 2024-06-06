import helper
import numpy as np

M = 10  # Number of training samples
TRAIN_FILE_PATH = 'C:/Users/user/Downloads/MSINT CSV/mnist_train.csv'

# Read the data from the CSV file
labels, activation0 = helper.ReadCSVFile(TRAIN_FILE_PATH, M)

# Forward propagation through the first layer
def First_Layer(W, X, b, M):
    Z = np.zeros((M, W.shape[1]))
    activations = np.zeros((M, W.shape[1]))

    for i in range(M):
        Z[i] = helper.Z(W, X[i], b)
        if np.max(Z[i]) != 0:
            Z[i] = np.divide(Z[i], np.max(Z[i]))
        activations[i] = helper.BigReLu(Z[i])
    if dropout:
        activations, mask = helper.DropoutForward(activations, dropout_rate)
        return Z, activations, mask
    else:
        return Z, activations
        


def OutputLayer(W, A1, b, M):
    Z = np.zeros((M, W.shape[0]))
    activations = np.zeros((M, W.shape[0]))

    for i in range(M):
        Z[i] = helper.Z(W, A1[i], b)
        if np.max(Z[i]) != 0:
            Z[i] = np.divide(Z[i], np.max(Z[i]))
        activations[i] = helper.BigReLu(Z[i])
    return Z, activations

# Loss
def TotalLoss(output, lambd, M):
    return 2 * np.mean(np.subtract(helper.CorrectOutput(labels, M), output) ** 2) + (lambd / 2) * (np.sum(weights_1 ** 2) + np.sum(weights_2 ** 2))

# Back Propagation
def ErrorInOutputLayer(zazas, M):
    return AllEOL(zazas, M).mean(axis=0)

def AllEOL(zazas,M):
    S = np.subtract(zazas[3], helper.CorrectOutput(labels, M))
    for row in range(np.shape(zazas[2])[0]):
        for i in range(np.shape(zazas[2])[1]):
            if zazas[2][row][i] == 0:
                S[row][i] = 0
    return S

def ErrorInFirstLayer(zazas, M):
    return AllEFL(zazas, M).mean(axis=0)

def AllEFL(zazas, M):
    S = AllEOL(zazas, M) @ weights_2

    for row in range(np.shape(zazas[0])[0]):
        for i in range(np.shape(zazas[0])[1]):
            if zazas[0][row][i] == 0:
                S[row][i] = 0
    if dropout:
        return S * mask
    else:
        return S

def CostWithBias(zazas, layer, M):
    if layer == 1:
        return np.mean(ErrorInFirstLayer(zazas, M))
    elif layer == 2:
        return np.mean(ErrorInOutputLayer(zazas, M))
    return None

def CostWithWeight(zazas ,layer, M):
    if layer == 1:
        S = np.zeros(np.shape(weights_1))
        E = AllEFL(zazas, M)
        for i in range(M):
            S = np.add(S, np.outer(E[i], activation0[i]))
        return S / M
    elif layer == 2:
        S = np.zeros(np.shape(weights_2))
        E = AllEOL(zazas ,M)
        for i in range(M):
            S = np.add(S, np.outer(E[i], zazas[1][i]))
        return S / M
    return None

# Gradients
def CostGradient(zazas, lambd, M):
    global weights_1, weights_2
    W1 = CostWithWeight(zazas, 1, M).reshape(-1) + lambd * weights_1.reshape(-1)
    W2 = CostWithWeight(zazas, 2, M).reshape(-1) + lambd * weights_2.reshape(-1)
    B1 = CostWithBias(zazas, 1, M)
    B2 = CostWithBias(zazas, 2, M)

    return np.concatenate((W1, W2, [B1, B2]))

def Theta():
    return np.concatenate((weights_1.reshape(-1), weights_2.reshape(-1), [bias_1, bias_2]))

def GradientDescent(STEP_SIZE, lambd, zazas, M):
    theta = Theta()
    theta += -STEP_SIZE * RMSprop(zazas, lambd, M)

    w1 = theta[:614656].reshape((784, 784))
    w2 = theta[614656:622496].reshape((10, 784))
    b1 = theta[-2]
    b2 = theta[-1]
    return w1, w2, b1, b2

def RMSprop(zazas, lambd, M):
    global E
    g = CostGradient(zazas, lambd, M)
    E = decay_rate * E + (1 - decay_rate) * np.square(g)
    return g / (np.sqrt(E + 1e-8))


def AverageFirstInputs(X, W1, W2, B1, B2, K):
    tempW1 = np.zeros(weights_1.shape)
    tempW2 = np.zeros(weights_2.shape)
    tempB1 = tempB2 = 0.0
    for i in range(K):
        W1T = W1
        W2T = W2
        B1T = B1
        B2T = B2
        for j in range(10):
            x = np.array([X[i]])
            z1, a1 = First_Layer(W1T, x, B1T, 1)
            z2, a2 = OutputLayer(W2T, a1, B2T, 1)
            zaza = helper.CreateZaza(z1, a1, z2, a2)
            W1T, W2T, B1T, B2T = GradientDescent(step_size, regularization, zaza, 1)
        tempW1 += W1T
        tempW2 += W2T
        tempB1 += B1T
        tempB2 += B2T
    tempW1 = (tempW1 / K).reshape(weights_1.shape)
    tempW2 = (tempW2 / K).reshape(weights_2.shape)
    tempB1 = tempB1 / K
    tempB2 = tempB2 / K
    return tempW1, tempW2, tempB1, tempB2, True

# Weights and Biases initializations
weights_1 = helper.Xavier_initialization((784, 784))
weights_2 = helper.Xavier_initialization((10, 784))
bias_1 = bias_2 = 1.0 
regularization = 1e-3 
dropout_rate = 0.2 # doesnt do shit still overfits

decay_rate = 0.9
E = np.zeros(622498)
step_size = 0.00001


dropout = False
prevloss = float('inf')
# Training loop
labels, activation0 = helper.ShuffleTrainingSamples(labels, activation0)
weights_1, weights_2, bias_1, bias_2, dropout= AverageFirstInputs(activation0, weights_1, weights_2, bias_1, bias_2, 10)
for i in range(101):
    labels, activation0 = helper.ShuffleTrainingSamples(labels, activation0)
    z1, a1, mask = First_Layer(weights_1, activation0, bias_1, M)
    z2, a2 = OutputLayer(weights_2, a1, bias_2, M)
    zaza = helper.CreateZaza(z1, a1, z2, a2)

    weights_1, weights_2, bias_1, bias_2 = GradientDescent(step_size, regularization, zaza, M)


    loss = TotalLoss(a2, regularization, M)
    
    labels, activation0 = helper.ShuffleTrainingSamples(labels, activation0) # doesn't do shit lovely
    print(f"Loss difference = {np.abs(loss - prevloss)}")
    if (np.abs(loss - prevloss) > 0.2) and i > 1: # why 1 you ask? idk lol
        a2 = a2PREV
        print(f"Broke at Iteration = {i}")
        break
    else:
        prevloss = loss
        a2PREV = a2

    if i % 1 == 0:
        print(f"Iteration = {i}")
        print(f"{labels[0]} = {a2[0]}")
        print(f"{labels[1]} = {a2[1]}")
        print("---------------------")

print(f"Labels = {labels}")
print(f"Neural Network Outputs = {helper.OutputAsLabel(a2, M)}")

