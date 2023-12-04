# code for HMM classification on 38D mfcc data. 
# PRML A3 - Vikram CS19B021, Vedant Saboo CS19B074

import numpy as np
import os
import matplotlib.pyplot as plt
import shutil
from hmmmodule import *
from plotmodule import *

DIM = 38                                                    # dim of data
colors = ['red','brown','lime','gold','dodgerblue']         # for plotting comparative ROC/DET curves

# code to extract data from file. this is assumed to be in the proper format per specs.
def extractData(file):
    lines = file.readlines()
    l0 = lines[0].strip().split(' ')
    _, num_vectors = int(l0[0]), int(l0[1])     # entry 2 of line 1 = number of feature vectors
    # loop to get all the feature vectors
    vectors = []
    for i in range(1,num_vectors+1):
        l = lines[i]
        l = l.strip().split(' ')
        l = np.array([np.double(l[j]) for j in range(0,DIM)],dtype=np.double)
        vectors.append(l)
    return np.array(vectors)                    # return the mfcc data as a numpy 2D array

# code to get name of folder [FOLDER] with the data and the class names
isodigitsfolder = str(input('Enter the name of the folder containing the isolated digits data : '))
l = input('Enter the class names separated by spaces : ')
classnames = l.strip().split()
numclasses = len(classnames)

# code to extract training data from FOLDER/cname/train/ and build trainData dictionary
# trainData[c] = list of templates from class c where each template is a 2-tuple = (template's file name, template's data)
trainData = dict()
for cname in classnames:
    templates = []
    path = isodigitsfolder + '/' + cname + '/train/'
    for fname in os.listdir(path):
        if(fname.__contains__('.mfcc')):
            with open(os.path.join(path,fname),'r') as f:
                templates.append((fname,extractData(f)))
    trainData[cname] = templates

# code to extract dev data from FOLDER/cname/dev/ and build dev list
# devData[i] = list of 3-tuples where each tuple D = (true class of D, file name of D, data of D)
devData = []
for cname in classnames:
    path = isodigitsfolder + '/' + cname + '/dev/'
    for fname in os.listdir(path):
        if(fname.__contains__('.mfcc')):
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
outfolder = str(input('Enter a name X. The plots will be saved in X_ISDHMM_Results: ')) + '_ISDHMM_Results'
if os.path.exists(outfolder):
    shutil.rmtree(outfolder)
os.mkdir(outfolder)

# get the number of different cases the user wants to try
print('Enter the number of cases (at most 5):')
ncases = int(input())
if(ncases > 5):
    exit(-1)

# get the state counts(classwise) and symbol count for each case
stateslist, symcntlist = [], []
for i in range(0,ncases):
    states = dict()
    symcntlist.append(int(input('Enter the M value for case {} : '.format(i+1))))
    l = input('Enter the N values for classes (in the order {}) for case {} separated by spaces : '.format(classnames,i+1))
    l = l.strip().split()
    l = list(map(int,l))
    for j in range(0,numclasses):
        states[classnames[j]] = l[j]
    stateslist.append(states)

fig1, roc = plt.subplots(figsize=(15,15), num=1)    # figure for ROC
fig2, det = plt.subplots(figsize=(15,15), num=2)    # figure for DET

# loop for running the different cases
for cs in range(0,ncases):
    # M_HMM = Number of symbols of HMM and states = dict from class c to number of states in hmm model for c
    states, M_HMM = stateslist[cs], symcntlist[cs]

    # generate vector quantisation codebook for HMM using KMeans clustering
    kmeans = constructCodebook(trainData,M_HMM,0)

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
        generateHMMFile(cname+'.hmm',trainDataObsSeq[cname],states[cname],M_HMM)

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

    # code to save the confusion matrix and plot ROC/DET curves on resp. axes for the current case
    statelbl =  '{'+'|'.join(cname + ':' + str(states[cname]) for cname in classnames)+'}'
    plt.savefig(outfolder+'/ConfusionMatrix_N={}_M={}.png'.format(statelbl,M_HMM))
    roc = ROC_curve(truth=truthlist,scorelists=scorelists,ax=roc,clr=colors[cs],lbl='N={}_M={}'.format(statelbl,M_HMM))
    det = DET_curve(truth=truthlist,scorelists=scorelists,ax=det,clr=colors[cs],lbl='N={}_M={}'.format(statelbl,M_HMM))

# save the ROC/DET curves in the outfolder
plt.figure(num=1)
roc.legend()
roc.set_title(' Receiver Operating Characteristic ')
plt.savefig(outfolder+'/ReceiverOperatingCharacteristic.png')
plt.figure(num=2)
det.legend()
det.set_title(' Detection Error Tradeoff ')
plt.savefig(outfolder+'/DetectionErrorTradeoff.png')

# save the last case's hmm model files in outfolder 
for c in classnames:
    os.system('mv '+c+'.hmm '+outfolder+'/')

# for optional part
# save the last case's codebook in outfolder
savecodebook(kmeans,'kmeans')
os.system('mv kmeans.codebook '+outfolder+'/kmeans.codebook')