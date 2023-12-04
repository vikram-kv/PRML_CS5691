# code for DTW on 2D handwritten character dataset. performs DTW without normalizing the data set
# PRML A3 - Vikram CS19B021, Vedant Saboo CS19B074

import numpy as np
import os
import shutil 
from plotmodule import *
from dtwmodule import findScore
import sys

DIM = 2                                                     # dim of data
colors = ['red','brown','lime','gold','dodgerblue']         # for plotting comparative ROC/DET curves

# check for proper usage. the first cmdline arg is YES if window(s) must be used in DTW and NO otherwise
# second cmdline arg = number of least errors to average in order to get the score of a test vector against a class
argc = len(sys.argv)
if(argc != 3 or ((str(sys.argv[1]) != 'YES') and (str(sys.argv[1]) != 'NO'))):
    print('Usage: python3 [scriptname] [YES|NO] [TOP_N] where YES|NO is for use a window or not, TOP_N = number of lowest scores to consider for averaging')
    exit(-1)

# there are 69 templates per class. perform a sanity check on TOP_N
TOP_N = int(sys.argv[2])
if(TOP_N > 69):
    exit(-1)

# code to extract data from file. this is assumed to be in the proper format per specs.
def extractData(file):
    lines = file.readlines()
    ln = lines[0].strip().split(' ')
    num_vectors = int(ln[0])        # entry 1 = num of points = n
    # subsequent entries are x[1] y[1] x[2] y[2] ... x[n] y[n] where x[i] and y[i] are coordinates of point i
    ln = ln[1:]
    vectors = []
    for i in range(0,num_vectors):
        l = np.array([np.double(ln[2*i]),np.double(ln[2*i+1])],dtype=np.double)
        vectors.append(l)
    return np.array(vectors)        # return the character data as a numpy 2D array

# code to get input from user about name of folder [FOLDER] with the data, number of classes and class names
handcharfolder = str(input('Enter the name of the folder containing the handwritten char data : '))
numclasses = int(input('Enter the number of classes(characters): '))
classnames = []

print('Enter the class names: ')
for i in range(0,numclasses):
    cname = str(input())
    classnames.append(cname)

# code to extract training data from FOLDER/train/ and build trainData dictionary
# trainData[c] = list of templates from class c where each template is a 2-tuple = (template's file name, template's data)
trainData = dict()
for cname in classnames:
    templates = []
    path = handcharfolder + '/' + cname + '/train/'
    for fname in os.listdir(path):
        if(fname.__contains__('.txt')):
            with open(os.path.join(path,fname),'r') as f:
                templates.append((fname,extractData(f)))
    trainData[cname] = templates

# code to extract dev data from FOLDER/dev/ and build dev list
# devData[i] = list of 3-tuples where each tuple D = (true class of D, file name of D, data of D)
devData = []
for cname in classnames:
    path = handcharfolder + '/' + cname + '/dev/'
    for fname in os.listdir(path):
        if(fname.__contains__('.txt')):
            with open(os.path.join(path,fname),'r') as f:
                data = extractData(f)
                devData.append((cname,fname,data))

# check if user wants to use windows or not
ch = str(sys.argv[1])
if ch == 'YES':
    # we allow only 5 windows or else the comparative ROC/DET plot will become clumsy
    # get number of diff window sizes the user wants
    print("Enter the number of window sizes (at most 5).")
    print("Suggestion : Values should be in [1-20] for good results.")
    wsizeslen = int(input())
    if(wsizeslen > 5):
        exit(-1)
    wsizes = []
    # get the window sizes
    print("Enter one-sided window sizes.")
    for i in range(0,wsizeslen):
        wsizes.append(int(input()))
else:
    # dummy placeholders to ensure the computation loop runs once
    # as ch is NO, there wont be any window used
    wsizeslen = 1
    wsizes = [0]

# get a prefix name for the folder under which we save the plots
outfolder = str(input('Enter a string N. The plots will be saved in N_HCDDTW_Unnorm_Plots: ')) + "_HCDDTW_Unnorm_Plots"
if os.path.exists(outfolder):
    shutil.rmtree(outfolder)
os.mkdir(outfolder)

fig1, roc = plt.subplots(figsize=(15,15), num=1)    # figure for ROC
fig2, det = plt.subplots(figsize=(15,15), num=2)    # figure for DET

# loop for running with diff window sizes
for i in range(0,wsizeslen):
    wsize = wsizes[i]
    scorelists = dict()
    for cname in classnames:
        scorelists[cname] = []

    # truthlist[i], predlist[i] = true class of dev data[i], pred class of dev data[i]
    # pred class = class with lowest DTW score    
    truthlist, predlist = [], []
    for trueclass, _, test in devData:
        pred = None
        minscore = np.PINF
        for c in classnames:
            cscore = findScore(trainData[c],test,TOP_N,ch,wsize)    # wsize is used only if ch is YES
            scorelists[c].append(cscore)
            if(cscore < minscore):
                pred = c
                minscore = cscore
        predlist.append(pred)
        truthlist.append(trueclass)
    
    # for plotting the ROC and DET curves, we negate the scores so that the best score
    # becomes the highest score (this is required to use our module)
    for cname in classnames:
        scorelists[cname] = np.array(scorelists[cname],dtype=np.double)
        scorelists[cname] *= -1

    _, cfm = plt.subplots(figsize=(15,15),num=(i+3))
    cfm = ConfusionMatrixPlotter(classlist=classnames,truth=truthlist,pred=predlist,ax=cfm)

    if (ch == 'YES'):
        # code to save the confusion matrix and plot ROC/DET curves on resp. axes for this window size
        plt.savefig(outfolder+'/ConfusionMatrix_wsize={}.png'.format(wsize))
        roc = ROC_curve(truth=truthlist,scorelists=scorelists,ax=roc,clr=colors[i],lbl='wsize={}'.format(wsize))
        det = DET_curve(truth=truthlist,scorelists=scorelists,ax=det,clr=colors[i],lbl='wsize={}'.format(wsize))
    else:
        # code to save the confusion matrix and plot ROC/DET curves on resp. axes for DTW with no window
        plt.savefig(outfolder+'/ConfusionMatrix.png')
        roc = ROC_curve(truth=truthlist,scorelists=scorelists,ax=roc,clr=colors[i],lbl='No window'.format(wsize))
        det = DET_curve(truth=truthlist,scorelists=scorelists,ax=det,clr=colors[i],lbl='No window'.format(wsize))

# save the ROC/DET curves in the outfolder
plt.figure(num=1)
roc.legend()
roc.set_title('Receiver Operating Characteristic')
plt.savefig(outfolder+'/ReceiverOperatingCharacteristic.png')
plt.figure(num=2)
det.legend()
det.set_title('Detection Error Tradeoff')
plt.savefig(outfolder+'/DetectionErrorTradeoff.png')