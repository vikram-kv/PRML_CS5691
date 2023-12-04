import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import DetCurveDisplay

def plotCharacter(featurevectors):
    # code for plotting cfm, cfm1; labeling the axes; saving the plot
    _, ax = plt.subplots(figsize=(10, 10))
    xs = [f[0] for f in featurevectors]
    ys = [f[1] for f in featurevectors]
    ax.plot(xs,ys)
    plt.title('Character', fontsize=20)
    plt.show()

def ConfusionMatrixPlotter(classlist : list, truth : list, pred : list, ax : matplotlib.axes.Axes):
    # cfm - unnormalized confusion matrix
    # cfm1 - normalized confusion matrix
    clcount = len(classlist)
    cfm = np.zeros((clcount,clcount),dtype=np.int32)
    cfm1 = np.zeros((clcount,clcount),dtype=np.float64)
    _count = len(truth)
    
    classindexer = dict()
    for i in range(0,len(classlist)):
        classindexer[classlist[i]] = i

    # cfm convention - columns : truth, rows : prediction
    # code to build cfm, cfm1
    for i in range(0,_count):
        cfm[classindexer[pred[i]]][classindexer[truth[i]]] += 1
    for i in range(0,len(classlist)):
        for j in range(0,len(classlist)):
            cfm1[i,j] = cfm[i][j]/_count

    ax.matshow(cfm)
    for i in range(cfm.shape[0]):
        for j in range(cfm.shape[1]):
            ax.text(x=j, y=i,s=str(cfm[i, j])+'\n\n'+'{:.2%}'.format(cfm1[i,j]), va='center', ha='center', size='xx-large',color='w')
    
    ax.set_xlabel('True Class', fontsize=15)
    ax.set_ylabel('Predicted Class', fontsize=15)
    ax.set_title('Confusion Matrix', fontsize=20)
    ax.set_xticks(np.arange(clcount)), ax.set_yticks(np.arange(clcount))
    ax.set_xticklabels(classlist), ax.set_yticklabels(classlist)
    return ax

def constructThresholdList(scorelists : dict):
    l = []
    for cname in scorelists.keys():
        l += scorelists[cname].tolist()
    l.sort(key = lambda x : x, reverse=True)
    return l

def ROC_curve(truth : list, scorelists : dict, ax : matplotlib.axes.Axes, clr, lbl):

    # formatting the roc plot by adding title, labels, etc.
    ax.set_xlim([0.0, 1.05]), ax.set_ylim([0.0, 1.05])
    ax.plot([0, 1], [0, 1], color='black', lw=2, linestyle="--")
    ax.set_xlabel("False Positive Rate"), ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC")

    N = len(truth)
    thresholds = constructThresholdList(scorelists)     # thresholds = all score values
    tpr, fpr = [0.0], [0.0]                             # init tpr, fpr
    for th in thresholds:
        TP, FP = 0, 0
        TN, FN = 0, 0
        for cname in scorelists.keys():
            slist = scorelists[cname]
            for i in range(0,N):
                if(slist[i] > th):                      # this elem's score is above th
                    if(cname == truth[i]):              # if scorer class = true class of elem, then we have a True Positive. Otherwise, we have a False Positive. 
                        TP += 1
                    else:
                        FP += 1
                else:                                   # this elem's score is below th
                    if(cname == truth[i]):              # if scorer class = true class of elem, then we have a False Negative. Otherwise, we have a True Negative.
                        FN += 1
                    else:
                        TN += 1
        tpr.append(TP/(TP+FN))
        fpr.append(FP/(FP+TN))
    tpr, fpr = np.asarray(tpr,dtype=np.float64), np.asarray(fpr,dtype=np.float64)
    ax.plot(fpr,tpr,label=lbl,color=clr)
    return ax


def DET_curve(truth : list, scorelists : dict, ax : matplotlib.axes.Axes, clr, lbl):

    N = len(truth)
    thresholds = constructThresholdList(scorelists) # thresholds = all score values
    fpr, fnr = [0.0], [1.0]                         # init tpr, fpr
    for th in thresholds:
        TP, FP = 0, 0
        TN, FN = 0, 0
        for cname in scorelists.keys():
            slist = scorelists[cname]
            for i in range(0,N):
                if(slist[i] > th):                  # this elem's score is above th
                    if(cname == truth[i]):          # if scorer class = true class of elem, then we have a True Positive. Otherwise, we have a False Positive. 
                        TP += 1
                    else:
                        FP += 1
                else:                               # this elem's score is below th
                    if(cname == truth[i]):          # if scorer class = true class of elem, then we have a False Negative. Otherwise, we have a True Negative.
                        FN += 1
                    else:
                        TN += 1
        fpr.append(FP/(FP+TN))
        fnr.append(FN/(TP+FN))
    fpr, fnr = np.asarray(fpr,dtype=np.float64), np.asarray(fnr,dtype=np.float64)
    
    display = DetCurveDisplay(fpr=fpr,fnr=fnr)
    display.plot(ax,color=clr,label=lbl)
    return ax