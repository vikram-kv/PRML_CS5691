'''
Module - PDFPlot.py
----------------------
3D Display of PDFs of each class along with contour projection on the x1-x2 plane
-------------------------------------------
Author -- Vikram - CS19B021, Vedant - CS19B074
Team   -- Team 6
Course -- CS 5691 Pattern Recognition and Machine Learning
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as mpatches
from mpl_toolkits import mplot3d
from Helpers import GridHelper

# main plotting function
def PDFPlotter(_range,mus,covs):

    fig, ax = plt.subplots(figsize=(10, 10),subplot_kw={'projection':'3d'})

    X1, X2, Z = GridHelper(_range,400,mus,covs)
    peak = 1.5 * np.max(Z)

    # plotting of pdfs where pdf{i} = gaussian with mean mus[i], covariance covs[i]
    ax.plot_surface(X1, X2, Z[0], rstride=5, cstride=5, antialiased=True, linewidth=1, color='r',edgecolors='black',alpha=0.7)
    ax.plot_surface(X1, X2, Z[1], rstride=5, cstride=5, antialiased=True, linewidth=1, color='g',edgecolors='black',alpha=0.7)
    ax.plot_surface(X1, X2, Z[2], rstride=5, cstride=5, antialiased=True, linewidth=1, color='b',edgecolors='black',alpha=0.7)

    # plotting of contour projections for pdf{i}
    ax.contour(X1, X2, Z[0], zdir='z', offset=-peak, cmap=cm.viridis, levels=4)
    ax.contour(X1, X2, Z[1], zdir='z', offset=-peak, cmap=cm.viridis, levels=4)
    ax.contour(X1, X2, Z[2], zdir='z', offset=-peak, cmap=cm.viridis, levels=4)

    # adjust axis lengths and labelling rate for z axis
    ax.set_xlim(_range[0][0], _range[0][1])
    ax.set_ylim(_range[1][0], _range[1][1])
    ax.set_zlim(-peak,peak)
    ax.set_zticks(np.linspace(0,peak,8))

    # set axes labels and plot title
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title('PDFs with contour curves')

    # build legend; save the plot
    p1 = mpatches.Patch(color='r', label='Class 1')
    p2 = mpatches.Patch(color='g', label='Class 2')
    p3 = mpatches.Patch(color='b', label='Class 3')
    ax.legend(handles=[p1,p2,p3])
