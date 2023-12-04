import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from KMeans import KMeans

from scipy.stats import multivariate_normal
class GMM:
    def __init__(self, k:int, max_iter:int =5, max_k_iter:int =10, seed=7):
        self.k = k
        self.max_iter = max_iter 
        self.max_k_iter = max_k_iter
        self.seed = seed
        np.random.seed(seed)

    def initialize(self, X):
        # theta = [Mu, Sigma, Pi]
        n, d = X.shape
        ##################################################
        # USE K Means Engine to initialise (The M0 Step) #
        ##################################################
        kmeans = KMeans(k=self.k, max_iter= self.max_k_iter)
        centroids, (assignment,kprime) = kmeans.fit(X)
        self.mu = np.array(centroids, dtype=np.float64)
        self.sigma = np.array([ np.cov(X.T) for _ in range(self.k) ], dtype=np.float64)
        self.gamma = np.full((n,self.k),fill_value=(1/self.k), dtype=np.float64)
        self.pi = np.mean(self.gamma, axis=0, dtype=np.float64)
                
    def gaussian(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray):
        d = len(x)
        norm_factor = (2*np.pi)**d
        
        mean_sigma = np.mean(sigma)
        det = np.linalg.det(sigma / mean_sigma)
        det = det * (mean_sigma ** len(sigma))
        norm_factor *= det
        norm_factor = 1.0/np.sqrt(np.abs(norm_factor))

        x_mu = np.matrix(x-mu)
        rs = norm_factor*np.exp(-0.5*x_mu*np.linalg.inv(sigma)*x_mu.T)
            
        return rs

    # E-Step: 
    def e_step(self, X):
        n,d = X.shape
        likelihood = np.zeros( (n, self.k) ) 
        for i in range(self.k):
            likelihood[:,i] = multivariate_normal(mean=self.mu[i],cov=self.sigma[i], allow_singular=True, seed=self.seed).pdf(X)
        numerator = likelihood * self.pi
        denominator = numerator.sum(axis=1)[:, np.newaxis]
        self.gamma = numerator / denominator
        if(np.isnan(self.gamma).any()):
            self.gamma[np.isnan(self.gamma)] = 1/(self.k)
        self.pi = np.mean(self.gamma, axis=0)


    # M-Step: update meu and sigma holding phi and weights constant
    def m_step(self, X):
        for i in range(self.k):
            row_i = self.gamma[:, [i]]
            total_ = row_i.sum()

            self.mu[i] = (X * row_i).sum(axis=0) / total_
            self.sigma[i] = np.cov(X.T,aweights=(row_i/total_).flatten(), bias=True)
            
                

    # responsible for clustering the data points correctly
    def fit(self, X):
        # initialise parameters like weights, phi, meu, sigma of all Gaussians in dataset X
        self.initialize(X)
        for iteration in tqdm(range(self.max_iter),desc='GMM (EM): k='+str(self.k)):
            # iterate to update the value of P(Xi/Ci=j) and (phi)k
            self.e_step(X)
            # iterate to update the value of meu and sigma as the clusters shift
            self.m_step(X)
    
    # predict function 
    def predict(self, X):
        n,d = np.array(X).shape
        gamma = np.zeros((n,self.k))
        likelihood = np.zeros( (n, self.k) ) 
        for i in range(self.k):
            likelihood[:,i] = multivariate_normal(mean=self.mu[i],cov=self.sigma[i], allow_singular=True, seed=self.seed).pdf(X)
        numerator = likelihood * self.pi
        denominator = numerator.sum(axis=1)[:, np.newaxis]
        gamma = numerator / denominator
        if(np.isnan(gamma).any()):
            gamma[np.isnan(gamma)] = 1/(self.k)
        # self.pi = np.mean(self.gamma, axis=0)
        return np.argmax(gamma, axis=1)
    
    def score(self, x):
        # X is d-dimensional single input feature vector
        X = np.array([x])
        n, d = X.shape
        likelihood = np.zeros( (n, self.k) ) 
        for i in range(self.k):
            likelihood[:,i] = multivariate_normal(mean=self.mu[i],cov=self.sigma[i], allow_singular=True, seed=self.seed).pdf(X)
        numerator = likelihood * self.pi
        p_x = numerator.sum(axis=1)[:, np.newaxis]
        return p_x.item()
    
    def plot_contour(self, fig, ax, _range, _input, color='red'):
        N, _, d = _input.shape
        if(d != 2):
            print("Can't plot for input not 2d")
            return None
        Z = np.empty((self.k,N,N))
        for k in range(0,self.k):
            for i in range(N):
                for j in range(N):
                    Z[k][i][j] = GMM.gaussian(_input[i][j], self.mu[k], self.sigma[k])
            ax.contour(_input[:,:,0], _input[:,:,1], Z[k], colors=color, levels=np.arange(0.1, 2.0, 0.1))