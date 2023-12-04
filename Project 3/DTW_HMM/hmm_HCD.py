# code for HMM classification on 2D handwritten character data. 
# PRML A3 - Vikram CS19B021, Vedant Saboo CS19B074

import numpy as np
import os
import matplotlib.pyplot as plt
import shutil
from hmmmodule import *
from plotmodule import *

DIM = 2                                                     # dim of data
colors = ['red','brown','lime','gold','dodgerblue']         # for plotting comparative ROC/DET curves

# code for normalizing the character defined by points
def normalize(points):
    # maxX - maximum of x coordinates of all pts, minX = minimum of x coordinates of all points. for y, the vars are defined similarly
    minX, minY = np.PINF, np.PINF
    maxX, maxY = np.NINF, np.NINF
    for c in points:
        x, y = c[0], c[1]
        minX, minY = np.amin([minX,x]), np.amin([minY,y])
        maxX, maxY = np.amax([maxX,x]), np.amax([maxY,y])
    
    # code for finding the center's coordinates and making the center the origin by shifting all points
    cenX, cenY = (minX+maxX)/2, (minY+maxY)/2
    N = len(points)
    for i in range(0,N):
        points[i][0] -= cenX
        points[i][1] -= cenY
    
    # code for rescaling the character to a 1 x 1 figure (ideally). however, because most characters have diff
    # widths and heights, we take a common scale factor for both dimensions
    idH = idW = 1
    curH = maxY - minY
    curW = maxX - minX
    r = max(idH/curH,idW/curW)
    for i in range(0,N):
        points[i][0] *= r
        points[i][1] *= r

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
    normalize(vectors)              # normalize the character
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
                devData.append((cname,fname,extractData(f)))

# find the log of prior probabilities for each class
# prior[c] = (no of samples from c)/(total number of training samples)
prior = dict()
N_TRAIN = 0
for c in classnames:
    prior[c] = len(trainData[c])
    N_TRAIN += prior[c]
for k in prior.keys():
    prior[k] = np.log(prior[k]/N_TRAIN)

# get a prefix name for the folder under which we save the plots
outfolder = str(input('Enter a name X. The plots will be saved in X_HCDHMM_Plots: ')) + '_HCDHMM_Plots'
if os.path.exists(outfolder):
    shutil.rmtree(outfolder)
os.mkdir(outfolder)

# get the number of different (n,m) combinations the user wants to try
print('Enter the number of cases (different N-M combinations) (at most 5):')
ncases = int(input())
if(ncases > 5):
    exit(-1)
print('Enter the N and M values for each case. All values must be followed by a newline.')

# get the (n,m) values for each combination
statecntlist, symcntlist = [], []
for i in range(0,ncases):
    statecntlist.append(int(input()))
    symcntlist.append(int(input()))

fig1, roc = plt.subplots(figsize=(15,15), num=1)    # figure for ROC
fig2, det = plt.subplots(figsize=(15,15), num=2)    # figure for DET

# loop for running with diff (n,m) combinations
for cs in range(0,ncases):
    # N_HMM = Number of states of HMM and M_HMM = Number of symbols of HMM
    N_HMM, M_HMM = statecntlist[cs], symcntlist[cs]

    # generate vector quantisation codebook for HMM using KMeans clustering
    kmeans = constructCodebook(trainData,M_HMM,200)

    # trainDataObsSeq is the same as trainData but each vector in all train vector series
    # is replaced with the centroid index that the vector is closest to in the codebook
    trainDataObsSeq = dict()
    for cname in classnames:
        templates = []
        for fname, data in trainData[cname]:
            templates.append((fname,constructObsSequence(kmeans,data)))
        trainDataObsSeq[cname] = templates

    # devDataObsSeq is the same as devData but each vector in all dev vector series
    # is replaced with the centroid index that the vector is closest to in the codebook
    devDataObsSeq = []
    for cname, fname, data in devData:
        devDataObsSeq.append((cname,fname,constructObsSequence(kmeans,data)))

    # generate the model files for all classes with N_HMM states and M_HMM symbols
    for cname in classnames:
        generateHMMFile(cname+'.hmm',trainDataObsSeq[cname],N_HMM,M_HMM)

    # generate scorelists where scorelists[c][i] = log likelihood of the ith dev vector series
    # given class c + log prior of c = log posterior of c
    scorelists = dict()
    for cname in classnames:
        scorelists[cname] = testAgainstHMMFile(cname+'.hmm',devDataObsSeq)
        scorelists[cname] += prior[cname]

    # truthlist[i], predlist[i] = true class of dev data[i], pred class of dev data[i]
    # pred class = class with highest log posterior
    truthlist, predlist = [], []
    N_DEV = len(devData)
    for i in range(0,N_DEV):
        maxscore = np.NINF
        pred = None
        for c in classnames:
            cscore = scorelists[c][i]
            if(cscore > maxscore):
                pred = c
                maxscore = cscore
        predlist.append(pred)
        truthlist.append(devData[i][0])

    _, cfm = plt.subplots(figsize=(15,15),num=(cs+3))
    cfm = ConfusionMatrixPlotter(classlist=classnames,truth=truthlist,pred=predlist,ax=cfm)

    # code to save the confusion matrix and plot ROC/DET curves on resp. axes for the current (n,m) combination
    plt.savefig(outfolder+'/ConfusionMatrix_N={}_M={}.png'.format(N_HMM,M_HMM))
    roc = ROC_curve(truth=truthlist,scorelists=scorelists,ax=roc,clr=colors[cs],lbl='N={}_M={}'.format(N_HMM,M_HMM))
    det = DET_curve(truth=truthlist,scorelists=scorelists,ax=det,clr=colors[cs],lbl='N={}_M={}'.format(N_HMM,M_HMM))

# save the ROC/DET curves in the outfolder
plt.figure(num=1)
roc.legend()
plt.savefig(outfolder+'/ReceiverOperatingCharacteristic.png')
plt.figure(num=2)
det.legend()
plt.savefig(outfolder+'/DetectionErrorTradeoff.png')

# save the last case's hmm model files in outfolder
for c in classnames:
    os.system('mv '+c+'.hmm '+outfolder+'/')