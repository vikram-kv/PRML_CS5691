# code for KMeans clustering class to be used for vector quantisation in HMMs
# PRML A3 - Vikram CS19B021, Vedant Saboo CS19B074
import numpy as np

# class for the kmeans object
class KMeans:
    # k = number of clusters and max_iter = number of iterations to run for
    def __init__(self, k, max_iter):
        self.k = k
        self.max_iter = max_iter
    
    # initialize the centroids randomly from the input vectors X. use 'seed' to seed the random module
    # to ensure results are reproducible
    def initialize(self, X, seed):
        self.N, self.d = X.shape
        shuffled = np.copy(X)
        np.random.seed(seed) 
        np.random.shuffle(shuffled)
        self.centroids = shuffled[:self.k]          # list of centroids(means)
        self.assignment = np.zeros(self.N)          # assignment[i] = index of the cluster the ith vector is assigned to
    
    # function for expectation step
    def e_step(self, X):
        for i in range(self.N):                     # for all points
            x = X[i]
            scores = []
            # find the dist of each centroid from the current point and assign the current point to the closest centroid
            for k in range(self.k):
                scores.append(np.linalg.norm(x - self.centroids[k]))
            self.assignment[i] = np.argmin(scores)
    
    # function for m step
    def m_step(self, X):
        # recompute the centroid of cluster c as the mean of the points assigned to c
        for k in range(self.k):
            ind_k = np.where(self.assignment == k)
            X_k = X[ind_k]
            self.centroids[k] = np.mean(X_k, axis=0)

    # function to initialize the kmeans object with input points X and random module seed s
    # and do EM for max_iter iterations
    def fit(self, X, s):
        self.initialize(X, s)
        for _ in range(self.max_iter):
            self.e_step(X)
            self.m_step(X)
        return self
    
    # function to predict the cluster indices for the vectors in X
    def predict(self, X):
        N, _ = X.shape
        passignment = np.zeros(N,dtype=np.uint32)
        for i in range(N):
            x = X[i]
            scores = []
            # assign x to the closest centroid
            for k in range(self.k):
                scores.append(np.linalg.norm(x - self.centroids[k]))
            passignment[i] = np.argmin(scores)
        return passignment
