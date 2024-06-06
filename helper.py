import numpy as np
import csv
import xlsxwriter
np.random.seed(23)

def Kaiming_initialization(shape):
    std = np.sqrt(2 / shape[0])
    return np.random.randn(*shape) * std

def Xavier_initialization(shape):
    limit = np.sqrt(6 / (shape[0] + shape[1]))
    return np.random.uniform(-limit, limit, shape)

def ReLu(z):
    return max(0, z)

def BigReLu(z):
    return np.maximum(0, z)

def LeakyReLu(z, negative_slope=0.01):
    return np.maximum(0, z) + negative_slope * np.minimum(0, z)

def BigLeakyReLu(z, negative_slope=0.01):
    return np.where(z > 0, z, z * negative_slope)

def Z(weights, x, bias):
    return np.dot(weights, x) + bias

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

def ToExcel(x):
    name = f"test{np.shape(x)}.xlsx"
    workbook = xlsxwriter.Workbook(name)
    worksheet = workbook.add_worksheet()

    for row_num, row_data in enumerate(x):
        for col_num, cell_data in enumerate(row_data):
            worksheet.write(row_num, col_num, cell_data)

    workbook.close()

def DropoutForward(X, dropout_rate):
    mask = (np.random.rand(*X.shape) > dropout_rate).astype(np.float32) / (1 - dropout_rate)
    X_dropped = mask * X 
    return X_dropped, mask

def DropoutBackward(dout, mask):
    dX = dout * mask
    return dX

def ShuffleTrainingSamples(labels, values):
    combined = np.column_stack((labels, values))
    np.random.shuffle(combined)
    shuffled_labels, shuffled_values = combined[:, 0], combined[:, 1:]
    return shuffled_labels, shuffled_values


def CreateZaza(z1, a1, z2, a2):
    zaza = np.empty(4, dtype=object)
    zaza[0] = z1
    zaza[1] = a1
    zaza[2] = z2
    zaza[3] = a2
    return zaza