import helper1
import numpy as np
from tqdm import tqdm

TRAINING_SAMPLES = 10
FILE_PATH_TRAINING_SAMPLE = "C:/Users/user/Downloads/MSINT CSV/mnist_train.csv"

weights_1 = helper1.xavier_initialization((784,784))
weights_2 = helper1.xavier_initialization((10,784))
E = np.zeros(622497)

bias_1 = 1

# Constants for adam
b1 = 0.9; b2 = 0.999; e = 1e-8; t = 0
m = v = np.zeros((622497,))



labels , inputs = helper1.read_csv_file(FILE_PATH_TRAINING_SAMPLE, TRAINING_SAMPLES)


def first_layer():
    return helper1.layer(inputs, weights_1, bias_1)

def output_layer():
    return helper1.layer(activation_1, weights_2,activation="Sigmoid")

def cross_entropy():
    y = helper1.correct_output(labels, M=TRAINING_SAMPLES)
    error = np.zeros_like(output)
    for i in range(TRAINING_SAMPLES):
        error[i] = np.multiply(y[i], np.log(output[i]))
    return -error

def error_in_output_layer(): # ♥
    return np.dot(np.dot((helper1.correct_output(labels) / output).T, (activation_1 * ( 1 - activation_1))), inputs.T)

def error_in_first_layer(): # ♥
    return np.multiply(np.dot(weights_2.T, error_in_output_layer()), helper1.ReLU_derivative(z_1).T)

def cost_with_weight(layer):
    if layer == 1: # first layer
        return np.dot(error_in_first_layer(), inputs)
    elif layer == 2: # second layer
        return np.dot(error_in_output_layer(), activation_1)
    else:
        return None
    
def cost_with_bias():
    return np.mean(error_in_first_layer())

def cost_gradient():
    # takes all the costs and flattens them into an array for easy computation
    w1 = cost_with_weight(1).flatten()
    w2 = cost_with_weight(2).flatten()
    return np.concatenate((w1, w2, cost_with_bias().flatten()))

def flatten_parameters():
    # flattens all the weights and biases
    return np.concatenate((weights_1.reshape(-1), weights_2.reshape(-1), np.array(bias_1).reshape(-1)))

def gradient_descent(LEARNING_RATE):
    theta = flatten_parameters()
    theta += - LEARNING_RATE * rms_prop()
    return helper1.repack_gradient(theta)

def rms_prop():
    global E
    g = cost_gradient()
    E = E * 0.9 + 0.1 * np.square(g)
    return g / (np.sqrt(E + 1e-8))

def output_to_label():
    guesses = np.zeros((TRAINING_SAMPLES, 1))
    for i in range(TRAINING_SAMPLES):
        guesses[i] = np.argmax(output[i])
    return guesses

def check_guesses(guesses):
    correct = 0
    for i in range(TRAINING_SAMPLES):
        if guesses[i] == labels[i]:
            correct += 1
    return correct / TRAINING_SAMPLES



def adam(LEARNING_RATE):
    global t, b1, b2, e, m, v
    t += 1
    g = cost_gradient()
    
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * (g ** 2)
    mhat = m / (1 - b1 ** t)
    vhat = v / (1 - b2 ** t)
    
    theta = flatten_parameters()
    theta += -LEARNING_RATE * mhat / (np.sqrt(vhat) + e)
    return helper1.repack_gradient(theta)







'''

activation_1, z_1 = first_layer()
output, z_out = output_layer()

#helper1.to_excel(error_in_output_layer(), "error_output")
'''





# Training
epochs = 1000
lowest_error = 1000
lowest_error_index = -1
i = 0

for i in tqdm(range(epochs)):
    activation_1, z_1 = first_layer()
    output, z_out = output_layer()
    weights_1, weights_2, bias_1 = adam(LEARNING_RATE=1e-7)

    error = cross_entropy()
    if lowest_error > np.sum(error):
        lowest_error = np.sum(error)
        lowest_error_index = i

    if i % 100 == 0:
            print("---------------------------")
            print(f"Output = {output[0]}")
            # print(f"Specific Error = {np.round(error, 3)}")
            print(f"Total error = {error}")
            print(f"\n\nGuesses = {output_to_label().T}")
            print(f"% Correct = {np.round(check_guesses(output_to_label()) * 100,2)}")
        # labels, inputs = helper1.shuffle_training_samples(labels, inputs)


#print(f"labels = {labels}")
#print(f"Lowest error = {lowest_error}")
#print(f"Lowest error index = {lowest_error_index}")

#helper1.to_excel(binary_cross_entropy(), "errors_after_shuffle_10")

