'''
Module - DCBPlot.py
----------------------
Display of decision boundaries, decision regions(a region where the posterior of one class is the largest),
development vectors (color-labeled by their true class; predicted class is found using the decision region 
in which a dev vector lies), and projection of contour curves on the plane. 
-------------------------------------------
Author -- Vikram - CS19B021, Vedant - CS19B074
Team   -- Team 6
Course -- CS 5691 Pattern Recognition and Machine Learning
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcol
from Helpers import GridHelper

# main plotting function
def DecisionBoundaryPlotter(_range,mus,covs,data,truth,classifier):

    fig, ax = plt.subplots(figsize=(10, 10))

    # construct grid and apply the three gaussians on it
    X1, X2, Z = GridHelper(_range,600,mus,covs)

    # plot contours
    ax.contour(X1, X2, Z[0], colors='red', levels=4)
    ax.contour(X1, X2, Z[1], colors='blue', levels=4)
    ax.contour(X1, X2, Z[2], colors='green', levels=4)

    # code to apply classifier function to each point on grid and use the result to plot 
    # decision regions
    _x1, _x2 = X1.flatten(), X2.flatten()
    surf = np.empty(_x1.shape)
    for i in range(0,_x1.size):
        surf[i] = classifier([_x1[i],_x2[i]])
    surf = surf.reshape(X1.shape)
    ax.contourf(X1, X2, surf,cmap = mcol.ListedColormap(['orange', 'yellow', 'violet']))

    # code to plot the dev vectors labeled by their true class
    c1x1, c1x2 = [], []
    c2x1, c2x2 = [], []
    c3x1, c3x2 = [], []
    _sz = len(data)
    for i in range(0,_sz):
        if truth[i] == 1:
            c1x1.append(data[i][0]), c1x2.append(data[i][1])
        elif truth[i] == 2:
            c2x1.append(data[i][0]), c2x2.append(data[i][1])
        else:
            c3x1.append(data[i][0]), c3x2.append(data[i][1])
    ax.scatter(c1x1, c1x2, edgecolors='red',label='class 1 data',s=20,facecolors='none')
    ax.scatter(c2x1, c2x2, edgecolors='blue',label='class 2 data',s=20,facecolors='none')
    ax.scatter(c3x1, c3x2, edgecolors='green',label='class 3 data',s=20,facecolors='none')

    # code to label axes and save the plot
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title('Decision Boundary Diagram')
    ax.legend()
