import numpy as np
import helper1
import matplotlib.pyplot as plt

FILE_PATH_TRAINING_SAMPLE = "C:/Users/user/Downloads/MSINT CSV/mnist_train.csv"

labels , input = helper1.read_csv_file(FILE_PATH_TRAINING_SAMPLE, 1000)

def draw_image(data):
    # Ensure the data is in the correct shape
    if data.shape != (28, 28):
        raise ValueError("Data must be of shape (784, 784)")
    
    # Display the image
    plt.imshow(data, cmap='gray')  # 'gray' colormap for grayscale images
    plt.axis('off')  # Turn off the axis
    plt.show()
'''
i = 0
ordered_inputs = np.zeros(10,784)
for x in range(1000):
    if labels[x] == i:
        ordered_inputs[i] = input[x]
        x += 1
    if x == 11:
        break
new_labels = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        '''
    
