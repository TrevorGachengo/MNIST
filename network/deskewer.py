import gzip
import pickle as cPickle
import numpy as np
import sys
from scipy.ndimage import affine_transform

'''
This file extracts the training and test data from the original mnist file and save them.
It will also save copies of the deskewed data. 
'''

def mnist(deskew):
    try:
        with gzip.open('data/mnist.pkl.gz', 'rb') as f:
            data = cPickle.load(f, encoding='bytes')
        f.close()
    except:
        print('\\data\\mnist.pkl.gz is not found please download the dataset.')
        sys.exit()
    
    (train_data, train_labels), (test_data, test_labels) = data

    train_data = reshape784(train_data)
    test_data = reshape784(test_data)

    if deskew:
        print("**********************")
        print("DESKEWING")
        train_data = deskewAll(train_data) / np.max(train_data)
        test_data = deskewAll(test_data) / np.max(test_data)
        print("DESKEWING COMPLETED")
        print("**********************")
    else:
        print("No deskewing")
        train_data = train_data / np.max(train_data)
        test_data = test_data / np.max(test_data)

    train_labels = train_labels.reshape((60000, 1))
    test_labels = test_labels.reshape((10000, 1))

    train = compact_data_labels(train_data, train_labels)
    test = compact_data_labels(test_data, test_labels)
    return train, test

def reshape784(x):
    return x.reshape((x.shape[0], 784))

def compact_data_labels(data, labels):
    '''
    Labels are last column of the returned array
    '''
    return np.concatenate((data, labels), axis=1)

def extract_data_labels(x):
    '''
    Returns: data, labels
    '''
    labels = x[:, 784]
    data = x[:, :-1]
    return data, labels

def moments(image):
    c0, c1 = np.mgrid[:image.shape[0], :image.shape[1]] # creates mesh grid

    totalImage = np.sum(image) # sum of pixels
    m0 = np.sum(c0 * image) / totalImage # mu_x
    m1 = np.sum(c1 * image) / totalImage # mu_y
    m00 = np.sum ((c0 - m0) **2 *image) / totalImage # var(x)
    m11 = np.sum((c1 - m1) ** 2 * image) / totalImage # var(y)
    m01 = np.sum((c0 - m0) *(c1 - m1) * image) / totalImage # covariance(x, y)
    mu_vector = np.array([m0, m1]) # mu_x and mu_y
    covariance_matrix = np.array([[m00, m01], [m01, m11]])
    return mu_vector, covariance_matrix

def deskew(image):
    c, v = moments(image)
    alpha = v[0, 1] / v[0, 0]
    affine = np.array([[1, 0], [alpha, 1]])
    ocenter = np.array(image.shape) / 2.0
    offset = c - np.dot(affine, ocenter)
    return affine_transform(image, affine, offset=offset)

def deskewAll(X):
    currents = []
    for i in range(len(X)):
        currents.append(deskew(X[i].reshape(28, 28)).flatten())
    return np.array(currents)

train, test = mnist(deskew=False)
np.save('network/data/train_data.npy', train)
np.save('network/data/test_data.npy', test)

train, test = mnist(deskew=True)
np.save('network/data/train_data_DSK.npy', train)
np.save('network/data/test_data_DSK.npy', test)
