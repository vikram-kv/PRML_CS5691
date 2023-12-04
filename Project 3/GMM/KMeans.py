import numpy as np
from tqdm import tqdm

class KMeans:
    '''
    K MEANS ENGINE
    ----------------------------------------------
    TAKES THE DATASET AS AN INPUT AND FITS HARD
    CLUSTERING USING THE EM ALGORITHM
    ----------------------------------------------
    NUMBER OF CLUSTERS AND ITERATIONS OF THE EM
    ALGORITHM HAVE TO BE SPECIFIED
    ----------------------------------------------
    '''
    def __init__(self, k:int, max_iter:int=10):
        self.k = k
        self.max_iter = max_iter
        
    def initialize(self, X):
        self.N, self.d = X.shape
        shuffled = np.copy(X)
        np.random.seed(6)
        np.random.shuffle(shuffled)
        self.centroids = shuffled[:self.k]
        self.assignment = np.zeros(self.N)
        
    def e_step(self, X):
        for i in range(self.N):
            x = X[i]
            scores = []
            for k in range(self.k):
                scores.append(np.linalg.norm(x - self.centroids[k]))
            self.assignment[i] = np.argmin(scores)
            
    def m_step(self, X):
        for k in range(self.k):
            ind_k = np.where(self.assignment == k)
            X_k = X[ind_k]
            self.centroids[k] = np.mean(X_k, axis=0)

    def fit(self, X):
        self.initialize(X)
        for _ in tqdm(range(self.max_iter),desc='K MEANS CLUSTERING: k='+str(self.k)):
            self.e_step(X)
            self.m_step(X)
        return self.centroids, self.predict(X)
            
    def predict(self, X):
        if(len(X.shape) == 1):
            X = np.array([X])
        N, _ = X.shape
        assignment = np.zeros(N,dtype=np.uint32)
        for i in range(N):
            x = X[i]
            scores = []
            for k in range(self.k):
                scores.append(np.linalg.norm(x - self.centroids[k]))
            assignment[i] = np.argmin(scores)
        # check for empty clusters!
        used = len(np.unique(assignment))
        return assignment, used