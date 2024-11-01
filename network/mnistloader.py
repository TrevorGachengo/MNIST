import numpy as np

def mnist(deskew=True):
    '''
    :desc Loads the MNIST dataset
    :param deskew: Returns deskewed data if true otherwise returns default MNIST data
    :return train, test
    The last column in each data is the labels
    '''
    if deskew:
        train = np.load('network/data/train_data_DSK.npy')
        test = np.load('network/data/test_data_DSK.npy')
    else:
        train = np.load('network/data/train_data.npy')
        test = np.load('network/data/test_data.npy')
    return train, test

