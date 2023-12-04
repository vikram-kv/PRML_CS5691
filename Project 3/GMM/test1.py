# %%
# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcol
from tqdm.notebook import tqdm_notebook as tqdm
import os
import csv

# %%
class KMeans:
    def __init__(self, k:int, max_iter:int=10):
        self.k = k
        self.max_iter = max_iter
        
    def initialize(self, X):
        self.N, self.d = X.shape
        shuffled = np.copy(X)
        np.random.seed(0)
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
        return assignment

# %%
class GMM:
    def __init__(self, k:int, max_iter:int =5, max_k_iter:int =10):
        self.k = k
        self.max_iter = max_iter 
        self.max_k_iter = max_k_iter

    def initialize(self, X):
        # theta = [Mu, Sigma, Pi]
        n, d = X.shape
        ##################################################
        # USE K Means Engine to initialise (The M0 Step) #
        ##################################################
        kmeans = KMeans(k=self.k, max_iter= self.max_k_iter)
        centroids, assignment = kmeans.fit(X)
        self.mu = np.array(centroids, dtype=np.float64)
        self.sigma = np.array([ np.cov(X.T) for _ in range(self.k) ], dtype=np.float64)
        self.gamma = np.zeros((n,self.k), dtype=np.float64)
        for i in range(n):
            self.gamma[i][assignment[i]] = 1
        self.pi = np.mean(self.gamma, axis=0, dtype=np.float64)
                
    def gaussian(x: np.ndarray, mu: np.ndarray, cov: np.ndarray):
            
        # return rs
        cov_det = np.linalg.det(cov)
        cov_inv = np.linalg.inv(cov)
        D = np.sqrt((2*np.pi)**2 * cov_det)
        # fac = np.einsum('k,kl,abl->ab', x-mu, cov_inv, x-mu)
        fac = (x - mu).T @ cov_inv @ (x-mu)
        return np.exp(-fac / 2) / D

    # E-Step: 
    def e_step(self, X):
        n,d = X.shape
        for i in range(n):
            x = X[i]
            p_x = 0.
            vec = []
            for k in range(self.k):
                p_xk = self.pi[k] * (GMM.gaussian(x, self.mu[k], self.sigma[k]))
                vec.append(p_xk.item())
                p_x += p_xk
            self.gamma[i] = (np.array(vec))*1.0/p_x
        self.pi = np.mean(self.gamma, axis=0)


    # M-Step: update meu and sigma holding phi and weights constant
    def m_step(self, X):
        for i in range(self.k):
            weight = self.gamma[:, [i]]
            total_weight = weight.sum()

            self.mu[i] = (X * weight).sum(axis=0) / total_weight
            print(weight)
            self.sigma[i] = np.cov(X.T,aweights=(weight/total_weight).flatten(), bias=True)

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
        for i in range(n):
            x = X[i]
            p_x = 0.
            vec = []
            for k in range(self.k):
                p_xk = self.pi[k] * (GMM.gaussian(x, self.mu[k], self.sigma[k]))
                vec.append(p_xk.item())
                p_x += p_xk
            # if(p_x < 1e-9):
            #     print("p_x is ", p_x)
            gamma[i] = (np.array(vec, dtype=np.float64))*1.0/p_x
        # self.pi = np.mean(self.gamma, axis=0)
        return np.argmax(gamma, axis=1)
        weights = self.predict_proba(X)
        # datapoint belongs to cluster with maximum probability
        # returns this value
        return np.argmax(weights, axis=1)
    
    def score(self, x, ret_gamma: bool = False):
        # x is d-dimensional single input feature vector
        if(ret_gamma):
            gamma = np.zeros(self.k)
            vec = []
        p_x = 0.
        for k in range(self.k):
            p_xk = self.pi[k] * (GMM.gaussian(x, self.mu[k], self.sigma[k]))
            if(ret_gamma):
                vec.append(p_xk.item())
            p_x += p_xk.item()
        if(ret_gamma):
            gamma = (np.array(vec))*1.0/p_x
            return p_x, gamma
        return p_x
    
    def plot_contour(self, fig, ax, _range, _input, color='red'):
        N, _, d = _input.shape
        if(d != 2):
            print("Can't plot for input not 2d")
            return None
        Z = np.empty((self.k,N,N))
        print("Plotting contours...")
        for k in tqdm(range(0,self.k)):
            for i in range(N):
                for j in range(N):
                    Z[k][i][j] = GMM.gaussian(_input[i][j], self.mu[k], self.sigma[k])
            ax.contour(_input[:,:,0], _input[:,:,1], Z[k], colors=color, levels=np.arange(0.2, 2.0, 0.5))
        print("Done")

# %%
X = []
Y = []
classes = os.listdir("Features/")
for i in range(len(classes)):
    _class = classes[i]
    inputfiles = os.listdir("Features/"+_class+"/train/")
    for inputfile in inputfiles:
        with open("Features/"+_class+"/train/" + inputfile) as f:
            reader = np.genfromtxt(f, delimiter=" ")
            for row in reader:
                X.append(row)
                Y.append(i)
                
df = pd.DataFrame(X)
df.head()

# %%
Y = np.array(Y, dtype=np.float64)
X = np.array(X, dtype=np.float64)
ind1 = np.where(Y == 0)
print(ind1)
ind2 = np.where(Y == 1)
print(ind2)
ind3 = np.where(Y == 2)
print(ind3)
ind4 = np.where(Y == 3)
print(ind4)
ind5 = np.where(Y == 4)
print(ind5)

X1 = X[ind1]
X2 = X[ind2]
X3 = X[ind3]
X4 = X[ind4]
X5 = X[ind5]

# %%
gmm1 = GMM(8, 10, 30)
gmm1.fit(X1)

# %%
gmm2 = GMM(8, 10, 30)
gmm2.fit(X2)

# %%
gmm3 = GMM(8, 10, 30)
gmm3.fit(X2)

# %%
gmm4 = GMM(8, 10, 30)
gmm4.fit(X2)

# %%
gmm5 = GMM(8, 10, 30)
gmm5.fit(X2)

# %%
import numpy as np
from sklearn.mixture import GaussianMixture
# Suppose Data X is a 2-D Numpy array (One apple has two features, size and flavor)
gmm1 = GaussianMixture(n_components=4, random_state=0).fit(X1)
gmm2 = GaussianMixture(n_components=4, random_state=0).fit(X2)
gmm3 = GaussianMixture(n_components=4, random_state=0).fit(X3)
gmm4 = GaussianMixture(n_components=4, random_state=0).fit(X4)
gmm5 = GaussianMixture(n_components=4, random_state=0).fit(X5)

# %%
X_dev = []
Y_dev = []
classes = os.listdir("Features/")
for i in range(len(classes)):
    _class = classes[i]
    inputfiles = os.listdir("Features/"+_class+"/dev/")
    for inputfile in inputfiles:
        with open("Features/"+_class+"/dev/" + inputfile) as f:
            reader = np.genfromtxt(f, delimiter=" ")
            for row in reader:
                X_dev.append(row)
                Y_dev.append(i)

# %%
post1 = lambda x: gmm1.score(x)
post2 = lambda x: gmm2.score(x)
post3 = lambda x: gmm3.score(x)
post4 = lambda x: gmm4.score(x)
post5 = lambda x: gmm5.score(x)
classifier = lambda x : np.argmax([post1(x),post2(x),post3(x),post4(x),post5(x)])

# %%
# construct a meshgrid using _range. N = no of sampling points(common for both x1, x2 intervals)
# returns the meshgrid(X1, X2) and Z = result of applying the 3 pdfs(pdf1 with mean mus[0], 
# covariance covs[0] and so on) on the meshgrid
def GridHelper(_range,N,mus,covs):
    X1 = np.linspace(_range[0][0], _range[0][1], N)
    X2 = np.linspace(_range[1][0], _range[1][1], N)
    X1, X2 = np.meshgrid(X1, X2)

    _input = np.empty((N,N,2))
    _input[:,:,0] = X1
    _input[:,:,1] = X2

    return (X1,X2,_input)

# %%
# main plotting function
import matplotlib.colors as mcol
from tqdm.notebook import tqdm_notebook as tqdm
def DecisionBoundaryPlotter(_range,mus,covs,data,truth,classifier):

    fig, ax = plt.subplots(figsize=(10, 10))

    # construct grid and apply the three gaussians on it
    print("Making Grid...")
    X1, X2, _input = GridHelper(_range,100,mus,covs)
    print("Done.")

    # plot contours
    gmm1.plot_contour(fig,ax,_range,_input,'red')
    gmm2.plot_contour(fig,ax,_range,_input,'pink')

    # code to apply classifier function to each point on grid and use the result to plot 
    # decision regions
    _x1, _x2 = X1.flatten(), X2.flatten()
    surf = np.empty(_x1.shape)
    print("Evaluating classifier for ", _x1.size, " points...")
    for i in tqdm(range(0,_x1.size)):
        surf[i] = classifier([_x1[i],_x2[i]])
    print("Done.")
    surf = surf.reshape(X1.shape)
    ax.contourf(X1, X2, surf,cmap = mcol.ListedColormap(['orange', 'violet']))

    # code to plot the dev vectors labeled by their true class
    c1x1, c1x2 = [], []
    c2x1, c2x2 = [], []
    _sz = len(data)
    for i in range(0,_sz):
        if truth[i] == 1:
            c1x1.append(data[i][0]), c1x2.append(data[i][1])
        elif truth[i] == 2:
            c2x1.append(data[i][0]), c2x2.append(data[i][1])
    print("Plotting dev data points...")
    ax.scatter(c1x1, c1x2, edgecolors='yellow',label='Class 1 data',s=20,facecolors='none')
    ax.scatter(c2x1, c2x2, edgecolors='blue',label='Class 2 data',s=20,facecolors='none')
    print("Done.")

    # code to label axes and save the plot
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title('Decision Boundary Diagram')
    ax.legend()
    plt.show()

# %%
DecisionBoundaryPlotter([[-16,-2],[-7,8]], [gmm1.mu,gmm2.mu], [gmm1.sigma,gmm2.sigma], X_dev, Y_dev, classifier=classifier)

# %%
def measure_accuracy_syn(gmm1: GMM, gmm2: GMM, X_dev,Y_dev):
    accuracy = 0.
    for i in range(len(X_dev)):
        x = X_dev[i]
        s1 = gmm1.score(x)
        s2 = gmm2.score(x)
        if(s1 > s2):
            if(Y_dev[i] == 1):
                accuracy += 1
            else:
                accuracy -= 1
        elif(s2 > s1):
            if(Y_dev[i] == 2):
                accuracy += 1
            else:
                accuracy -= 1
        else:
            print("Ambigious point")
            # random
            r = np.random.randint(1,3)
            if(r == Y_dev[i]):
                accuracy += 1
            else:
                accuracy -= 1
    accuracy *= 100/len(X_dev)
    return accuracy

# %%
def measure_accuracy(classifier, X_dev,Y_dev):
    accuracy = 0.
    for i in range(len(X_dev)):
        x = X_dev[i]
        pred = classifier(x)
        if(pred == Y_dev[i]):
            accuracy += 1
    accuracy *= 100/len(X_dev)
    return accuracy

# %%
print(measure_accuracy(classifier, X, Y))
print(measure_accuracy(classifier, X_dev, Y_dev))


