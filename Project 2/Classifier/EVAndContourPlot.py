'''
Module - EVAndContourPlot.py
----------------------
Display of and projection of contour curves on the plane and the eigenvectors of each class's covariance
matrix
-------------------------------------------
Author -- Vikram - CS19B021, Vedant - CS19B074
Team   -- Team 6
Course -- CS 5691 Pattern Recognition and Machine Learning
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from Helpers import GridHelper

# function to plot contours of gaussian pdfs(pdf i has mean mus[i] and covariance covs[i])
# along with the eigenvectors of the covariance matrices
def ConstantDensityAndEigenvectorPlotter(covs,mus,_range):
    fig, ax = plt.subplots(figsize=(10, 10))

    X1, X2, Z = GridHelper(_range,1000,mus,covs)
    ax.contour(X1, X2, Z[0], cmap='cool')
    ax.contour(X1, X2, Z[1], cmap='cool')
    ax.contour(X1, X2, Z[2], cmap='cool')
    EigenvectorPlotter(covs[0],mus[0],ax,'red',_range,'Class 1')
    EigenvectorPlotter(covs[1],mus[1],ax,'blue',_range,'Class 2')
    EigenvectorPlotter(covs[2],mus[2],ax,'green',_range,'Class 3')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title('Eigenvectors And Constant Density Curves')
    ax.legend()

# function to plot, with color = 'c' and label = 'lbl', the eigenvector of matrix 'M'
# centered at 'mu' on the axes 'ax'. '_range' is used to determining an appropriate length
def EigenvectorPlotter(M,mu,ax,c,_range,lbl):
    w, v = np.linalg.eig(M)
    size = np.sqrt((_range[0][1] - _range[0][0])*(_range[1][1]-_range[1][0]))/4
    scale = np.linspace(-size,size,10)
    points1 = np.empty((10,2))
    points2 = np.empty((10,2))
    for i in range(0,10):
        points1[i] = scale[i] * v[:,0] + mu
        points2[i] = scale[i] * v[:,1] + mu
    ax.plot(points1[:,0],points1[:,1],color=c,label=lbl)
    ax.plot(points2[:,0],points2[:,1],color=c)
