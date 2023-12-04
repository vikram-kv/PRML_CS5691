'''
Module - ROCDETPlot.py
----------------------
Case-comparative display of ROC and DET plots
-------------------------------------------
Author -- Vikram - CS19B021, Vedant - CS19B074
Team   -- Team 6
Course -- CS 5691 Pattern Recognition and Machine Learning
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# scorers = list of class posteriors. 
# for each class i, data item j, a tuple {scorers[i](data[j])=score of data[j] wrt class i, i+1=scorer class, 
# truth[j]=true class of data[j]} is added to a list l. the list is then sorted on the score in descending 
# order and it is returned
def constructScoreList(data,truth,scorers):
    l = []
    n = data.shape[0]
    for i in range(0,3):
        for j in range(0,n):
            l += [(scorers[i](data[j]),i+1,truth[j])]
    l.sort(key = lambda x : x[0],reverse=True)
    return l

def ROCDET_curves(data,truth,ccsm):

    ticker = mtick.PercentFormatter(xmax=1, decimals=None, symbol='%')
    fig, axs = plt.subplots(1,2,figsize=(20,20))

    # roc = left plot on axs[0] and det = right plot on axs[1]
    roc, det = axs[0], axs[1]

    # formatting the roc plot by adding title, labels, etc.
    roc.set_xlim([0.0, 1.05]), roc.set_ylim([0.0, 1.05])
    roc.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    roc.set_xlabel("False Positive Rate"), roc.set_ylabel("True Positive Rate")
    roc.set_title("ROC")

    # formatting the det curve by adding title, labels, etc.
    det.set_xlim([0.0, 1.05]), det.set_ylim([0.0, 1.05])
    det.xaxis.set_major_formatter(ticker), det.yaxis.set_major_formatter(ticker)
    det.set_xlabel("False Alarm Rate (in percentage) "), det.set_ylabel("Missed Detection Rate (in percentage) ")
    det.set_title("DET")

    # colors and labels for different cases
    colors = ['red','brown','lime','gold','dodgerblue']
    labels = ['Case '+str(i) for i in range(1,6)]
    cases = range(0,5)
    
    for _cse in cases:
        # get the scorers for case = _cse and construct the score list 'scores'
        scores = constructScoreList(data, truth, [ccsm[0][_cse], ccsm[1][_cse], ccsm[2][_cse]])
        # thresholds = all score values
        thresholds = [s[0] for s in scores]
        # init tpr, fnr, fpr
        tpr, fnr, fpr = [0.0], [1.0], [0.0]
        for th in thresholds:
            TP, FP = 0, 0
            TN, FN = 0, 0
            for elem in scores:
                if(elem[0] > th):
                    # this elem's score is above threhold th
                    if(elem[1] == elem[2]): # if scorer class = true class of elem, then we have a True Positive. Otherwise, we have a False Positive. 
                        TP += 1
                    else:
                        FP += 1
                else:
                    # this elem's score is below th
                    if(elem[1] == elem[2]): # if scorer class = true class of elem, then we have a False Negative. Otherwise, we have a True Negative.
                        FN += 1
                    else:
                        TN += 1
            tpr.append(TP/(TP+FN))
            fpr.append(FP/(FP+TN))
            fnr.append(FN/(TP+FN))
        # build numpy arrays for tpr, fpr, fnr and plot a curve on roc and det sections
        # with the corr. case color and case label
        tpr, fnr, fpr = np.asarray(tpr,dtype=np.float64), np.asarray(fnr,dtype=np.float64), np.asarray(fpr,dtype=np.float64)
        roc.plot(fpr,tpr,label=labels[_cse],color=colors[_cse])
        det.plot(fpr,fnr,label=labels[_cse],color=colors[_cse])
    
    # plot legends and save figure
    roc.legend(loc='lower right')
    det.legend()
