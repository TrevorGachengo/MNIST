import numpy as np
import csv
import xlsxwriter
np.random.seed(1)

def read_csv_file(file_path, M):
    with open(file_path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)  # Skip headers

        labels = []
        data = []
        for index, row in enumerate(csvreader):
            if index == M:
                break
            labels.append(int(row[0]))
            data.append([int(p) for p in row[1:]])
        return np.array(labels), np.divide(np.array(data), 255.0)

def xavier_initialization(shape):
    limit = np.sqrt(6 / (shape[0] + shape[1]))
    return np.random.uniform(-limit, limit, shape)

def Z(weights, x, bias=0):
    return np.dot(x, weights.T) + bias

def ReLU(z):
    return np.maximum(0, z)

def ReLU_derivative(z):
    return (z >= 0).astype(int)

def normalize(input):
    return np.divide(input, np.max(input))

def layer(input, weights, bias=0, activation="ReLU"):
    training_samples = input.shape[0]
    z = np.zeros((training_samples, weights.shape[0]))

    for i in range(training_samples):
        z[i] = Z(weights, input[i], bias)
    if activation == "ReLU":
        return ReLU(z), z
    elif activation == "Sigmoid":
        return sigmoid(z), z
    elif activation == "Softmax":
        return softmax(z), z

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    # a' = a * (1 - a)
    temp = sigmoid(z)
    return temp * (1 - temp)
    
def correct_output(labels, M=1):
    y = np.zeros((M, 10))
    for i in range(M):
        y[i, int(labels[i])] = 1
    return y

def to_excel(x, name):
    name = f"{name}.xlsx"
    workbook = xlsxwriter.Workbook(name)
    worksheet = workbook.add_worksheet()

    for row_num, row_data in enumerate(x):
        for col_num, cell_data in enumerate(row_data):
            worksheet.write(row_num, col_num, cell_data)

    workbook.close()

def shuffle_training_samples(labels, values):
    combined = np.column_stack((labels, values))
    np.random.shuffle(combined)
    shuffled_labels, shuffled_values = combined[:, 0], combined[:, 1:]
    return shuffled_labels, shuffled_values

def repack_gradient(x):
    weights_1 = x[:614656].reshape(784,784)
    weights_2 = x[614656:622496].reshape(10,784)
    bias_1 = x[-1]
    return weights_1, weights_2, bias_1

def square_elementwise(x):
    temp = np.zeros_like(x)
    for i in range(len(x)):
        temp[i] = x[i] ** 2
    return temp




def softmax(z):
    exponents = [np.exp(i) for i in z]
    return np.divide(exponents, np.sum(exponents))

def softmax_derrivative(Z):
    result = np.zeros((Z.shape[1],Z.shape[1]))
    eZj = np.sum(np.exp(Z),axis=0)
    for i in range(Z.shape[0]):
        eZi = np.exp(Z[i])
        result[i] = (eZi * (eZj - eZi)) / (eZj ** 2)
    return result
