# module that implements Principal Component Analysis (PCA) and Fischer Linear Discriminant Analysis (FDA) transformations
# PRML A4 - Vikram CS19B021, Vedant Saboo CS19B074

import numpy as np
from sklearn.covariance import empirical_covariance

'''
    class for performing PCA transformation
'''
class PCA:
    def __init__(self, n_comp):
        self.n_comp = n_comp
    
    '''
        X is the input data set as a numpy array where each row is a feature vector
    '''
    def fit(self, X):
        # shift all vectors by the overall mean and then compute covariance matrix for the shifted vectors
        X_mean = np.mean(X, axis=0)
        X_meanshifted = X - X_mean
        X_cov = np.cov(X_meanshifted, rowvar=False)

        # we know X_cov is real symmetric, so we use eigh function
        eigenvalues, eigenvectors = np.linalg.eigh(X_cov)

        # now, we sort the eigenvalues with eigenvectors in desc. order
        sorted = np.argsort(eigenvalues)[::-1]
        seigenvectors = eigenvectors[:,sorted]
        #update: this is unnecessary because eigh function sorts the eigenvalues in ascending order, and we just need to reverse the matrices
        #anyway, for safety and readability, we still write the sorting code

        # select the top n eigenvectors
        topn_eigenvectors = seigenvectors[:,0:self.n_comp]

        # get the transform matrix A
        self.A = topn_eigenvectors.T

    '''
        transform the data by representing each input vector using A
    '''
    def transform_data(self, data):
        return (self.A @ data.T).T

'''
    class for performing FDA transformation
'''
class FDA:
    '''
        X is the input data set where rows of X represent individual vectors
        classlabels is such that classlabels[i] is the class of X[i]
        red_dim is the number of dimensions after FDA reduction
    '''
    def __init__(self, X, classlabels, red_dim):
        self.data = X
        self.clbls = classlabels
        self.red_dim = red_dim
        self.dim = X.shape[1]
        self.N = X.shape[0]
        self.calculate_scatter_matrices()                       # calculate between class scatter Sb and with class scatter Sw
        self.calculate_eigenvalues_and_transformmatrix()        # solve the eigenvalue problem from inv(Sw) . Sb
    
    '''
        calculates Sb and Sw using their definitions.
        i)  Sw = sum of covariances of all classes with weight as number of train vectors from the given class
        ii) Sb = (N * total covariance) - Sw where N = number of train vectors
    '''
    def calculate_scatter_matrices(self):

        self.classes, indices = np.unique(self.clbls, return_inverse=True) # find the 
        clcount = np.bincount(indices)

        # calculation of Sw from definition
        Sw = np.zeros((self.dim,self.dim))
        for idx in range(len(self.classes)):
            cdata = self.data[indices == idx]
            Sw += clcount[idx] * empirical_covariance(cdata)
        self.Sw = Sw

        # calculation of Sb from definition
        St = empirical_covariance(self.data)
        self.Sb = self.N * St - Sw

    '''
        solve the eigenvalue problem from inv(Sw) . Sb and sort the eigenvectors (using 
        their corr. eigenvalues) in desc. order. do sanity check on red_dim -- it can't 
        be more than K - 1 where K = num of classes. The transformation matrix is also
        found by taking the top red_dim eigenvectors
    '''
    def calculate_eigenvalues_and_transformmatrix(self):

        eig_values,eig_vecs = np.linalg.eig(np.linalg.pinv(self.Sw) @ (self.Sb))
        sorted = np.argsort(eig_values)[::-1]
        self.seigenvectors = eig_vecs[:,sorted]
        self.red_dim = min(self.red_dim, len(self.classes) - 1)
        self.A = (self.seigenvectors[:,0:self.red_dim]).T
        self.A = np.real(self.A)

    '''
        transform the data by representing each input vector using A
    '''
    def transform_data(self, data):
        return (self.A @ data.T).T
