'''
Module - Helpers.py
----------------------
Some simple helper functions for mean, var estimation of both 1D and 2D data. All estimators are MLE-based.
Also has helper functions for applying gaussians on an input grid of points.
-------------------------------------------
Author -- Vikram - CS19B021, Vedant - CS19B074
Team   -- Team 6
Course -- CS 5691 Pattern Recognition and Machine Learning
'''
import numpy as np

# helper gaussian function. applies gaussian N(x|mu,cov) on the grid of points
# represented as inputs. inputs[a,b,0] = x1 and inputs[a,b,1] = x2 for the "ab"th 
# vector. the third dimension introduction is to allow the usage of einstein summation-based
# einsum() function.  
def GaussianOnInputMatrix(inputs, mu, cov):
    cov_det = np.linalg.det(cov)
    cov_inv = np.linalg.inv(cov)
    D = np.sqrt((2*np.pi)**2 * cov_det)
    fac = np.einsum('abk,kl,abl->ab', inputs-mu, cov_inv, inputs-mu)
    return np.exp(-fac / 2) / D

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

    Z = np.empty((3,N,N))
    for i in range(0,3):
        Z[i] = GaussianOnInputMatrix(_input,mus[i],covs[i])
    return (X1,X2,Z)

def sum2D(vtrs):
    _sum = np.zeros((2,),dtype=np.float64)
    for v in vtrs:
        _sum = _sum + v
    return _sum

# mean estimator of an array of 2D vectors
def mean2D(vtrs):
    N = vtrs.shape[0]
    return sum2D(vtrs)/N

# covariance estimator of an array of 2D vectors
def cov2D(vtrs):
    N = vtrs.shape[0]
    _sum = np.zeros((2,2),dtype=np.float64)
    mean = mean2D(vtrs)
    for v in vtrs:
        diff = v - mean
        _sum = _sum + np.outer(diff,diff)
    return (_sum/N)

# mean estimator of an array of 1D scalars
def mean1D(sclrs):
    _sum = 0.0
    N = sclrs.shape[0]
    for s in sclrs:
        _sum += s
    return ((_sum/N))

# variance estimator of an array of 1D scalars
def var1D(sclrs):
    _sum = 0.0
    N = sclrs.shape[0]
    mean = mean1D(sclrs)
    for s in sclrs:
        diff = s - mean
        _sum += diff * diff
    return ((_sum/N))

