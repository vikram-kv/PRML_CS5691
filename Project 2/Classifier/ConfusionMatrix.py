'''
Module - ConfusionMatrix.py
----------------------
Display of confusion matrix given the truth and prediction for development vectors
-------------------------------------------
Author -- Vikram - CS19B021, Vedant - CS19B074
Team   -- Team 6
Course -- CS 5691 Pattern Recognition and Machine Learning
'''
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

def ConfusionMatrixPlotter(truth,pred):
    # cfm - unnormalized confusion matrix
    # cfm1 - normalized confusion matrix
    cfm = np.zeros((3,3),dtype=np.int32)
    cfm1 = np.zeros((3,3),dtype=np.float64)
    _count = len(truth)
    
    # cfm convention - columns : truth, rows : prediction
    # code to build cfm, cfm1
    for i in range(0,_count):
        cfm[pred[i]-1][truth[i]-1] += 1
    for i in range(0,3):
        for j in range(0,3):
            cfm1[i,j] = cfm[i][j]/_count

    # code for plotting cfm, cfm1; labeling the axes; saving the plot
    fig, ax = plt.subplots(figsize=(10, 10))

    ax.matshow(cfm)
    for i in range(cfm.shape[0]):
        for j in range(cfm.shape[1]):
            ax.text(x=j, y=i,s=str(cfm[i, j])+'\n\n'+'{:.2%}'.format(cfm1[i,j]), va='center', ha='center', size='xx-large',color='w')
    
    plt.xlabel('Target Class', fontsize=15)
    plt.ylabel('Output Class', fontsize=15)
    plt.title('Confusion Matrix', fontsize=20)
    names = ['Class 1','Class 2','Class 3']
    ax.set_xticks(np.arange(len(names))), ax.set_yticks(np.arange(len(names)))
    ax.set_xticklabels(names), ax.set_yticklabels(names)
