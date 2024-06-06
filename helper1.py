import numpy as np
import csv

def Xavier_initialization(shape):
    limit = np.sqrt(6 / (shape[0] + shape[1]))
    return np.random.uniform(-limit, limit, shape)

def ReLu(z):
    return np.maximum(0, z)

def Softmax(zi, zj):
    return np.divide(np.exp(zi), np.sum(np.exp(zj),axis=0))

def DerrivativeSoftmax(Z):
    result = np.zeros((Z.shape[1],Z.shape[1]))
    eZj = np.sum(np.exp(Z),axis=0)
    for i in range(Z.shape[0]):
        eZi = np.exp(Z[i])
        result[i] = (eZi * (eZj - eZi)) / (eZj ** 2)
    return result

def Z(weights, x, bias):
    return np.dot(weights, x) + bias

def ReadCSVFile(file_path, M):
    with open(file_path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)  # Skip headers

        labels = []
        X = []
        for index, row in enumerate(csvreader):
            if index == M:
                break
            labels.append(int(row[0]))
            X.append([int(p) for p in row[1:]])
        return np.array(labels), np.divide(np.array(X), 255.0)
    
def CorrectOutput(labels, M):
    y = np.zeros((M, 10))
    for i in range(M):
        y[i, int(labels[i])] = 10
    return y

def OutputAsLabel(output ,M):
    outputL = np.zeros(M)
    for i in range(M):
        index = output[i].argmax()
        outputL[i] = index
    return outputL