import numpy as np
import pandas as pd
import csv
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcol
from GMM import GMM
from plotmodule import ROC_curve, DET_curve, ConfusionMatrixPlotter

def classifier(x: np.ndarray, *models):
    scores = []
    for model in models:
        scores.append(model.score(x))
    return np.argmax(scores)

def classifier_block(X: np.ndarray, *models):
    n_blocks = len(X)
    scores = []
    for x in X:
        row = []
        for model in models:
            s = model.score(x)
            row.append(s)
        scores.append(row)
    scores = np.array(scores)
    scores = np.sum(scores, axis=0)
    return np.argmax(scores)

def measure_accuracy(classifier, X_dev,Y_dev):
    accuracy = 0.
    for i in range(len(X_dev)):
        x = X_dev[i]
        pred = classifier(x)
        if(pred == Y_dev[i]):
            accuracy += 1
    accuracy *= 100/len(X_dev)
    return accuracy

def task1():
    # io train
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
    df_norm = (df - df.mean()) / (df.std())
    X = df_norm.to_numpy(dtype=np.float64)
    Y = np.array(Y, dtype=np.float64)
    
    # io dev
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
                    
    df1 = pd.DataFrame(X_dev)
    df_norm1 = (df1 - df1.mean()) / (df1.std())
    X_dev = df_norm1.to_numpy(dtype=np.float64)
    Y_dev = np.array(Y_dev, dtype=np.float64)
    
    ind1 = np.where(Y == 0)
    ind2 = np.where(Y == 1)
    ind3 = np.where(Y == 2)
    ind4 = np.where(Y == 3)
    ind5 = np.where(Y == 4)

    X1 = X[ind1]
    X2 = X[ind2]
    X3 = X[ind3]
    X4 = X[ind4]
    X5 = X[ind5]
    
    # construct model(S)
    gmm1 = GMM(k=8,max_iter=20,max_k_iter=30)
    gmm2 = GMM(k=8,max_iter=20,max_k_iter=30)
    gmm3 = GMM(k=8,max_iter=20,max_k_iter=30)
    gmm4 = GMM(k=8,max_iter=20,max_k_iter=30)
    gmm5 = GMM(k=8,max_iter=20,max_k_iter=30)
    
    # fit
    gmm1.fit(X1)
    gmm2.fit(X2)
    gmm3.fit(X3)
    gmm4.fit(X4)
    gmm5.fit(X5)
    
    # measure accuracy
    # accuracy = 0.
    # blocks = X_dev.reshape(-1,36,23)
    # for blocki in range(len(blocks)):
    #     block = blocks[blocki]
    #     pred = classifier_block(np.array(block), gmm1, gmm2, gmm3, gmm4, gmm5)
    #     if(pred == int(Y_dev[36*blocki])):
    #         accuracy += 1
    # print("Accuracy on dev data = ", accuracy / len(blocks))
    accuracy = measure_accuracy(classifier=lambda x: classifier(x, gmm1, gmm2, gmm3, gmm4, gmm5), X_dev=X_dev, Y_dev=Y_dev)
    print("Accuracy on dev data = ", accuracy)
    
    # plot graphs
    scorelists = {}
    i = 0
    for model in [gmm1, gmm2, gmm3, gmm4, gmm5]:
        row = []
        for x in X_dev:
            row.append(model.score(x))
        scorelists[classes[i]] = np.array(row)
        i += 1
    truth = []
    for i in range(len (Y_dev)):
        truth.append(classes[int(Y_dev[i])])
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title("ROC CURVE, clusters = " + str(8))
    ROC_curve(truth, scorelists, ax,'blue', 'ROC')
    plt.savefig('roc_gmm_image.jpeg')
    plt.close()
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title("DET CURVE, clusters = " + str(8))
    DET_curve(truth, scorelists, ax,'red', 'DET')
    plt.savefig('det_gmm_image.jpeg')
    plt.close()
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title("Confusion Matrix, clusters = " + str(8))
    pred = []
    for x in X_dev:
        y = classifier(x, gmm1, gmm2, gmm3, gmm4, gmm5)
        pred.append(classes[y])
    ConfusionMatrixPlotter(classes, truth, pred, ax)
    plt.savefig('cfm_gmm_image.jpeg')
    plt.close()

def task2(k):
    # io train
    X = []
    Y = []
    with open("6/train.txt") as f:
        reader = csv.reader(f)
        for row in reader:
            X.append([float(row[0]), float(row[1])])
            Y.append(int(row[2]))
                    
    # df = pd.DataFrame(X)
    # df_norm = (df - df.min()) / (df.max() - df.min())
    # X = df_norm.to_numpy(dtype=np.float64)
    X = np.array(X, dtype=np.float64)
    Y = np.array(Y, dtype=np.float64)
    
    # io dev
    X_dev = []
    Y_dev = []
    with open("6/dev.txt") as f:
        reader = csv.reader(f)
        for row in reader:
            X_dev.append([float(row[0]), float(row[1])])
            Y_dev.append(int(row[2]))
                    
    # df1 = pd.DataFrame(X_dev)
    # df_norm1 = (df1 - df1.min()) / (df1.max() - df1.min())
    # X_dev = df_norm1.to_numpy(dtype=np.float64)
    X_dev = np.array(X_dev, dtype=np.float64)
    Y_dev = np.array(Y_dev, dtype=np.float64)
    
    Y = np.array(Y)
    X = np.array(X)
    ind1 = np.where(Y == 1)
    ind2 = np.where(Y == 2)

    X1 = X[ind1]
    X2 = X[ind2]
    
    # construct model(S)
    gmm1 = GMM(k=k,max_iter=20,max_k_iter=30)
    gmm2 = GMM(k=k,max_iter=20,max_k_iter=30)
    
    # fit
    gmm1.fit(X1)
    gmm2.fit(X2)
    
    # measure accuracy
    accuracy = measure_accuracy(classifier=lambda x: classifier(x, gmm1, gmm2), X_dev=X_dev, Y_dev=Y_dev-1)
    print("Accuracy on dev data = ", accuracy)
    
    # plot graphs
    scorelists = {}
    i = 0
    for model in [gmm1, gmm2]:
        row = []
        for j in range(len(X_dev)):
            x = X_dev[j]
            row.append(model.score(x))
        scorelists[str(i+1)] = np.array(row)
        i += 1
    truth = []
    for i in range(len (Y_dev)):
        truth.append(str(int(Y_dev[i])))
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title("ROC CURVE, clusters = " + str(8))
    ROC_curve(truth, scorelists, ax,'blue', 'ROC')
    plt.savefig('roc_gmm_syn_'+str(k)+'.jpeg')
    plt.close()
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title("DET CURVE, clusters = " + str(8))
    DET_curve(truth, scorelists, ax,'red', 'DET')
    plt.savefig('det_gmm_syn_'+str(k)+'.jpeg')
    plt.close()
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title("Confusion Matrix, clusters = " + str(8))
    pred = []
    for x in X_dev:
        y = classifier(x, gmm1, gmm2)
        pred.append(str(y+1))
    ConfusionMatrixPlotter(['1','2'], truth, pred,ax)
    plt.savefig('cfm_gmm_syn_'+str(k)+'.jpeg')
    plt.close()
    
    # dcb
    fig, ax = plt.subplots(figsize=(10, 10))

    # construct grid and apply the three gaussians on it
    N = 100
    _range = [[-16,-2],[-7,8]]
    X1 = np.linspace(_range[0][0], _range[0][1], N)
    X2 = np.linspace(_range[1][0], _range[1][1], N)
    X1, X2 = np.meshgrid(X1, X2)

    _input = np.empty((N,N,2))
    _input[:,:,0] = X1
    _input[:,:,1] = X2

    # plot contours
    gmm1.plot_contour(fig,ax,_range,_input,'red')
    gmm2.plot_contour(fig,ax,_range,_input,'pink')

    # code to apply classifier function to each point on grid and use the result to plot 
    # decision regions
    _x1, _x2 = X1.flatten(), X2.flatten()
    surf = np.empty(_x1.shape)
    for i in range(0,_x1.size):
        surf[i] = classifier([_x1[i],_x2[i]], gmm1, gmm2) + 1
    surf = surf.reshape(X1.shape)
    ax.contourf(X1, X2, surf,cmap = mcol.ListedColormap(['orange', 'violet']))

    # code to plot the dev vectors labeled by their true class
    c1x1, c1x2 = [], []
    c2x1, c2x2 = [], []
    _sz = len(X_dev)
    for i in range(0,_sz):
        if Y_dev[i] == 1:
            c1x1.append(X_dev[i][0]), c1x2.append(X_dev[i][1])
        elif Y_dev[i] == 2:
            c2x1.append(X_dev[i][0]), c2x2.append(X_dev[i][1])
    ax.scatter(c1x1, c1x2, edgecolors='yellow',label='Class 1 data',s=20,facecolors='none')
    ax.scatter(c2x1, c2x2, edgecolors='blue',label='Class 2 data',s=20,facecolors='none')

    # code to label axes and save the plot
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title('Decision Boundary Diagram')
    ax.legend()
    plt.savefig('dcb_gmm_syn_'+str(k)+'.jpeg')
    plt.close()

if (__name__ == '__main__'):
    dataset = input("image/synthetic?")
    if(dataset == 'image'):
        task1()
    else:
        task2(16)
    