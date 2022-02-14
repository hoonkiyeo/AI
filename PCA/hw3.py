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
    Lambda, U = eigh(S, subset_by_index = [len(S) - m, len(S) - 1])
    return np.diag(Lambda[::-1]), np.fliplr(U)

def get_eig_prop(S, prop):
    #total sum of eigenvalues
    total_value = sum(eigh(S, eigvals_only = True))
    #minimum eigenvalue
    min_val = total_value * prop
    Lambda, U = eigh(S, subset_by_value = [min_val, np.inf])
    return np.diag(Lambda[::-1]), np.fliplr(U)
    
def project_image(image, U):
    return (np.transpose(image)@U)@np.transpose(U)

def display_image(orig, proj):
    orig = np.transpose(np.reshape(orig, (32,32)))
    proj = np.transpose(np.reshape(proj, (32,32)))
    
    fig, (ax1,ax2) = plt.subplots(1,2)
    ax1.set_title("Original")
    ax2.set_title("Projection")
    orig_img = ax1.imshow(orig, aspect = "equal")
    proj_img = ax2.imshow(proj, aspect = "equal")
    
    #set size of the figures
    fig.colorbar(orig_img, ax=ax1)
    fig.colorbar(proj_img, ax=ax2)
    fig.set_size_inches(10, 3.5)

    plt.show()
