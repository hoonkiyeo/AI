from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt

def load_and_center_dataset(filename):
    x = np.load(filename)
    n = len(x)
    d = len(x[0])
    recenter = x - np.mean(x, axis=0)
    return recenter

def get_covariance(dataset):
    trans_data = np.transpose(dataset)
    return (1 / (len(dataset) - 1)) * (trans_data@dataset)

def get_eig(S, m):
    # Your implementation goes here!
    pass

def get_eig_prop(S, prop):
    # Your implementation goes here!
    pass

def project_image(image, U):
    # Your implementation goes here!
    pass

def display_image(orig, proj):
    # Your implementation goes here!
    pass

print(load_and_center_dataset("YaleB_32x32.npy"))