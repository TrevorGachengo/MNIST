import numpy as np
np.random.seed(0)

class Network():
    def __init__(self, learning_rate=1e-4, mini_batch_size=50):
        self.weight_1 = kaiming_initialization(50, 784)
        self.weight_2 = kaiming_initialization(10, 50)
        self.bias_1 = np.zeros((self.weight_1.shape[0], 1))
        self.bias_2 = np.zeros((self.weight_2.shape[0], 1))
        

        self.mini_batch_size = mini_batch_size
        self.learning_rate = learning_rate

        # optimizer
        self.m = np.zeros_like(self.unpack_gradient(NABLAS=False))
        self.v = np.zeros_like(self.m)
        self.previous_accuracy = 0 # for my ghetto optimizer

    def hidden_layer(self, x):
        '''
        x.shape = (784, mini batch size)
        uses ReLU
        z1 = (50, mini batch size)
        a1 = (50, mini batch size)
        '''
        z1 = np.dot(self.weight_1, x) + self.bias_1
        a1 = ReLU(z1)
        return z1, a1
    
    def output_layer(self, a1):
        ''''
        a1.shape = (50, mini_batch_size)
        uses Sigmoid
        zout = (10, mini_batch_size)
        aout = (10, mini_batch_size)
        '''
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

    def train(self, train_data):
        '''
        train_data has to be of shape (784, size of data)
        '''
        np.random.shuffle(train_data)
        data, labels = extract_data_labels(train_data)

        for i in range(0, len(data), self.mini_batch_size): # loops through all mini_batches
            mini_batch = data[i: i + self.mini_batch_size].T # (784, mini batch size)
            mini_batch_labels = labels[i: i + self.mini_batch_size] # (mini batch size,)

            nwo, nbo, nwh, nbh = self.train_mini_batch(mini_batch, mini_batch_labels)
            self.update_parameters(nwo, nbo, nwh, nbh)
        
    def train_mini_batch(self, data, label):
        z1, a1, zout, aout = self.feed_forward(data)
        return self.backprop(z1, a1, zout, aout, data, label)
    
    def backprop(self, z1, a1, zout, aout, data, label):
        delta_output = self.cost_derivative_2(aout, label) * Sigmoid_derivative(zout) # (10, mini batch size)
        delta_hidden = np.dot(self.weight_2.T, delta_output) * ReLU_derivative(z1) # (50, mini batch size)

        # output layer
        nwo = np.dot(delta_output, a1.T) # (10, 50)
        nbo = np.mean(delta_output, axis=1, keepdims=True) # (10, 1)

        # hidden layer
        nwh = np.dot(delta_hidden, data.T) # (50, 784)
        nbh = np.mean(delta_hidden, axis=1, keepdims=True) # (50, 1)

        return nwo, nbo, nwh, nbh

    def adam(self, g):
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
        y = label_vector(label)
        return output - y
    
    def cost_derivative_2(self, output, label):
        y = self.label_array(label)
        return output - y
    
    def test(self, test):
        correct = 0
        data, labels = extract_data_labels(test)
        output = self.feed_forward(data.T)[-1].T

        for out, label in zip(output, labels):
            if np.argmax(out) == label:
                correct += 1
        return correct
    
    def use_network(self, input):
        return np.argmax(self.feed_forward(input)[-1])

    def epoch(self, epochs, train, test, display=False, fixed_learning_rate=True):
        '''
        :param epochs: Amount of epochs to train the network
        :param train: Training data
        :param test: Testing data
            Data should be in shape (number of samples, 785) where the 785th column is the labels
        :param display: To show the results as it trains
        '''
        for i in range(epochs):
            self.train(train)
            correct = self.test(test)

            current_accuracy = correct / len(test) * 100
            if not fixed_learning_rate:
                self.update_learning_rate(current_accuracy)
            self.previous_accuracy = current_accuracy

            if display:
                if i % np.ceil(epochs * 0.05) == 0 and i != 0:
                    print("Epoch {}\nAccuracy {:.2f}%".format(i, current_accuracy))
                    print(f"Learning rate is {self.learning_rate:.2E}")
                    print("------------------------------")
        print("Epoch {}\nAccuracy {:.2f}%".format(epochs, current_accuracy))
        print("------------------------------") 

    def update_learning_rate(self, current_accuracy):
            if abs(current_accuracy - self.previous_accuracy) < 0.5:
                self.learning_rate = self.learning_rate * 1.5

    def average_inputs(self, a, b, c, d):
        return a / self.mini_batch_size, b / self.mini_batch_size, c / self.mini_batch_size, d / self.mini_batch_size
    
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
        out = np.zeros((self.mini_batch_size, 10))
        for i, x in enumerate(label):
            out[i] = label_vector(x).reshape((10,))
        return out.T


    def save_parameters(self):
        np.save('network/data/trained_parameters/weights_1.npy', self.weight_1)
        np.save('network/data/trained_parameters/weights_2.npy', self.weight_2)
        np.save('network/data/trained_parameters/bias_1.npy', self.bias_1)
        np.save('network/data/trained_parameters/bias_2.npy', self.bias_2)

    def load_parameters(self):
        self.weight_1 = np.load('network/data/trained_parameters/weights_1.npy')
        self.weight_2 = np.load('network/data/trained_parameters/weights_2.npy')
        self.bias_1 = np.load('network/data/trained_parameters/bias_1.npy')
        self.bias_2 = np.load('network/data/trained_parameters/bias_2.npy')

        
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
    return zsig * (1 - zsig) * 100

def label_vector(label):
    out = np.zeros((10,1))
    out[int(label)] = 1
    return out

def kaiming_initialization(n_in, n_out):
    std_dev = np.sqrt(2.0 / n_in)
    weights = np.random.normal(0, std_dev, (n_in, n_out))
    return weights

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    y = np.tanh(x)
    return 1 - (y ** 2)

def extract_data_labels(x):
    '''
    Returns: data, labels
    '''
    labels = x[:, 784]
    data = x[:, :-1]
    return data, labels