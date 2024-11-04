import numpy as np

def ReLU(z):
    return np.maximum(0, z)

def ReLU_derivative(z):
    return np.where(z > 0, 1, 0)

def leaky_ReLU(z, alpha=0.01):
    return np.where(z > 0, z, alpha * z)

def leaky_ReLU_derivative(z, alpha=0.01):
    return np.where(z > 0, 1, alpha)

def Sigmoid(z):
    return 1 / (1 + np.exp(-z))

def Sigmoid_derivative(z):
    zsig = Sigmoid(z)
    return zsig * (1 - zsig)
