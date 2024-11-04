import numpy as np
import os
import sys
import time
from activations import ReLU, Sigmoid, ReLU_derivative, Sigmoid_derivative

class Network():
    def __init__(self, hidden_units=50, learning_rate=1e-4, batch_size=50):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        # initializations
        self.weight_1 = kaiming_initialization(hidden_units, 784)
        self.weight_2 = kaiming_initialization(10, hidden_units)
        self.bias_1 = np.zeros((self.weight_1.shape[0], 1))
        self.bias_2 = np.zeros((self.weight_2.shape[0], 1))

        # optimizers
        self.fixed_learning_rate = True
        self.m = np.zeros_like(self.unpack_gradient(NABLAS=False))
        self.v = np.zeros_like(self.m)
        self.previous_accuracy = 0


    def hidden_layer(self, x):
        z1 = np.dot(self.weight_1, x) + self.bias_1
        a1 = ReLU(z1)
        return z1, a1
    
    def output_layer(self, a1):
        zout = np.dot(self.weight_2, a1) + self.bias_2
        aout = Sigmoid(zout)
        return zout, aout
    
    def feed_forward(self, data):
        '''
        Returns: z1, a1, zout, aout
        '''
        z1, a1 = self.hidden_layer(data)
        zout, aout = self.output_layer(a1)
        return z1, a1, zout, aout
    
    def epoch(self, epochs, train, test, fixed_learning_rate=True, display=False):
        '''
        :param epochs: Amount of epochs to train the network
        :param train: Training data
        :param test: Testing data
            Data should be in shape (number of samples, 785) where the 785th column is the labels
        :param display: To show the results as it trains
        '''
        start = time.time()
        for i in range(epochs):
            self.train(train)
            correct = self.test(test)
            current_accuracy = correct / len(test) * 100

            if not fixed_learning_rate and current_accuracy < 95:
                self.update_learning_rate(current_accuracy)
            self.previous_accuracy = current_accuracy

            if display:
                if i % np.ceil(epochs * 0.05) == 0 and i != 0:
                    print("Epoch {}\nAccuracy {:.2f}%".format(i, current_accuracy))
                    print(f"Learning rate is {self.learning_rate:.2E}")
                    print("------------------------------")
            if i == 0:
                end = time.time()
                print("Each epoch will take {} seconds".format(int(end - start)))
        print("Epoch {}\nAccuracy {:.2f}%".format(epochs, current_accuracy))
        print("------------------------------") 

    def train(self, train_data):
        '''
        train_data has to be of shape (784, size of data)
        '''
        np.random.shuffle(train_data)
        data, labels = extract_data_labels(train_data)

        for i in range(0, len(data), self.batch_size): # loops through all mini_batches
            mini_batch = data[i: i + self.batch_size].T
            mini_batch_labels = labels[i: i + self.batch_size]

            nwo, nbo, nwh, nbh = self.train_mini_batch(mini_batch, mini_batch_labels)
            self.update_parameters(nwo, nbo, nwh, nbh)
        
    def train_mini_batch(self, data, label):
        z1, a1, zout, aout = self.feed_forward(data)
        return self.backprop(z1, a1, zout, aout, data, label)
    
    def backprop(self, z1, a1, zout, aout, data, label):
        delta_output = self.cost_derivative(aout, label) * Sigmoid_derivative(zout)
        delta_hidden = np.dot(self.weight_2.T, delta_output) * ReLU_derivative(z1)

        # output layer
        nwo = np.dot(delta_output, a1.T)
        nbo = np.mean(delta_output, axis=1, keepdims=True)

        # hidden layer
        nwh = np.dot(delta_hidden, data.T)
        nbh = np.mean(delta_hidden, axis=1, keepdims=True)

        return nwo, nbo, nwh, nbh

    def adam(self, g):
        if self.fixed_learning_rate:
            return g
        
        self.m = 0.9 * self.m + 0.1 * g
        self.v = 0.999 * self.v + 0.001 * np.square(g)

        mhat = self.m / (0.1)
        vhat = self.v / (0.001)
        return mhat / (np.sqrt(vhat) + 1e-8)

    def update_parameters(self, nwo, nbo, nwh, nbh):
        '''
        g is the gradient (all the nablas)
        theta is all the parameters (weights and biases)
        '''
        theta = self.unpack_gradient(NABLAS=False)
        g = self.unpack_gradient(nwo, nbo, nwh, nbh)

        theta = theta - self.learning_rate * self.adam(g)
        self.repack_gradient(theta)

    def cost_derivative(self, output, label):
        y = self.label_array(label)
        return output - y
    
    def test(self, test):
        '''
        :desc Test the network with test data
        :return Amount of correct guesses by the network
        '''
        correct = 0
        data, labels = extract_data_labels(test)
        output = self.feed_forward(data.T)[-1].T

        for out, label in zip(output, labels):
            if np.argmax(out) == label:
                correct += 1
        return correct
    
    def use_network(self, input):
        '''
        :desc For testing the network with a single input
        :return Networks guess
        '''
        return np.argmax(self.feed_forward(input)[-1])

    def update_learning_rate(self, current_accuracy):
        '''
        :desc Boosts the learning rate when the change in accuracy gets too low
        '''
        if abs(current_accuracy - self.previous_accuracy) < 0.5:
            self.learning_rate = self.learning_rate * 1.5

    def average_inputs(self, a, b, c, d):
        return a / self.batch_size, b / self.batch_size, c / self.batch_size, d / self.batch_size
    
    def unpack_gradient(self, wo=None, bo=None, wh=None, bh=None, NABLAS=True):
        '''
        wo = out[:self.weight_2.size]
        bo = out[self.weight_2.size:self.weight_2.size + self.bias_2.size]
        wh = out[-self.weight_1.size - self.bias_1.size: self.bias_1.size]
        bh = out[-self.bias_1.size:]
        '''
        if NABLAS:
            return np.concatenate((wo.flatten(), bo.flatten(), wh.flatten(), bh.flatten()))
        elif not NABLAS:
            return np.concatenate((self.weight_2.flatten(), self.bias_2.flatten(), self.weight_1.flatten(), self.bias_1.flatten()))
        
    def repack_gradient(self, g):
        self.weight_2 = g[:self.weight_2.size].reshape(self.weight_2.shape)
        self.bias_2 = g[self.weight_2.size:self.weight_2.size + self.bias_2.size].reshape(self.bias_2.shape)
        self.weight_1 = g[-self.weight_1.size - self.bias_1.size: -self.bias_1.size].reshape(self.weight_1.shape)
        self.bias_1 = g[-self.bias_1.size:].reshape(self.bias_1.shape)

    def label_array(self, label):
        out = np.zeros((label.shape[0], 10))
        for i, x in enumerate(label):
            out[i] = label_vector(x).reshape((10,))
        return out.T


    def save_parameters(self):
        create_missing_dir('data\\trained_parameters')
        np.save('data/trained_parameters/weights_1.npy', self.weight_1)
        np.save('data/trained_parameters/weights_2.npy', self.weight_2)
        np.save('data/trained_parameters/bias_1.npy', self.bias_1)
        np.save('data/trained_parameters/bias_2.npy', self.bias_2)

    def load_parameters(self):
        try:
            self.weight_1 = np.load('data/trained_parameters/weights_1.npy')
            self.weight_2 = np.load('data/trained_parameters/weights_2.npy')
            self.bias_1 = np.load('data/trained_parameters/bias_1.npy')
            self.bias_2 = np.load('data/trained_parameters/bias_2.npy')
        except:
            print('No saved files to be loaded into the network')
            sys.exit()



def label_vector(label):
    out = np.zeros((10,1))
    out[int(label)] = 1
    return out

def kaiming_initialization(n_in, n_out):
    std_dev = np.sqrt(2.0 / n_in)
    weights = np.random.normal(0, std_dev, (n_in, n_out))
    return weights

def extract_data_labels(x):
    '''
    Returns: data, labels
    '''
    labels = x[:, 784]
    data = x[:, :-1]
    return data, labels

def create_missing_dir(dir):
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    dir_to_check = os.path.join(curr_dir + '\\' + dir)
    if not os.path.isdir(dir_to_check):
        os.mkdir(dir_to_check)
