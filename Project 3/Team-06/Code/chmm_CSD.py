# code for optional part on connected spoken digit recognition
# PRML A3 - Vikram CS19B021 and Vedant Saboo CS19B074
import numpy as np
import os
DIM = 38

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

def extractCodebook(fname):
    with open(fname,'r') as f:
        lines = f.readlines()
        centroids = []
        for l in lines:
            l = l.strip().split()
            centroids.append(list(map(float,l)))
        return np.array(centroids)

# function to predict the cluster indices for the vectors in X given the centroids
def cenpredict(centroids, X):
    N, _ = X.shape
    passignment = np.zeros(N,dtype=np.uint32)
    M = len(centroids)
    for i in range(N):
        x = X[i]
        scores = []
        # assign x to the closest centroid
        for k in range(M):
            scores.append(np.linalg.norm(x - centroids[k]))
        passignment[i] = np.argmin(scores)
    return passignment

# convert the vector series vecSeq to the corr. symbol sequence, i.e, do vector quantisation
def constructObsSequence(centroids,vecSeq):
    obsSeq = cenpredict(centroids,vecSeq)
    return obsSeq

# test the dev observation sequences in 'dDataObsSeqs' against the hmm model file 'fname'
# returns the log likelihood that each observation symbol sequence came from model 'fname'
def testAgainstHMMFile(fname, dDataObsSeqs):
    # write all observation symbol sequences in temp.seq in the proper format
    with open('temp.seq','w') as f:
        for _, obsSeq in dDataObsSeqs:
            for o in obsSeq:
                f.write(str(o)+' ')
            f.write('\n')
    # test temp.seq against model fname
    os.system('./test_hmm temp.seq '+fname)
    # extract the log likelihoods of each observation symbol sequence from alphaout
    with open('alphaout','r') as f:
        lines = f.readlines()
        l = lines[0]
        l = l.strip().split()
        log_likelihoods = list(map(float,l))
    log_likelihoods = np.array(log_likelihoods,dtype=np.double)
    # remove the unnecessary files
    os.system('rm temp.seq alphaout')
    return log_likelihoods

# code to get input from user about name of folder [FOLDER] with the data, number of classes and class names
cisodigitsfolder = str(input('Enter the name of the folder containing the connected isolated digits data : '))
l = input('Enter the class names of isolated digits separated by spaces : ')
classnames = l.strip().split()
numclasses = len(classnames)

# code to extract dev data from FOLDER/dev/ and build dev list
# devData[i] = list of 2-tuples where each tuple D = (trueclass = file name of D, data of D)
devData = []
path = cisodigitsfolder + '/dev/'
for fname in os.listdir(path):
    if(fname.__contains__('.mfcc')):
        with open(os.path.join(path,fname),'r') as f:
            cname = fname.strip('.mfcc')
            devData.append((cname,extractData(f)))

# code to concatenate the HMMs by fixing the probability of inter-hmm transition as 0.5, i.e, the probability
# of going from the last state of current digit's hmm to the first state of the next digit's hmm is 0.5
hmm_data = [None for i in range(0,numclasses)]
orig_data = [None for i in range(0,numclasses)]
chn_data = [None for i in range(0,numclasses)]
states = [None for i in range(0,numclasses)]

# code to extract the data from the class's hmm model files and build orig_data = original model file data (used for the last hmm in 3-digit model),
# chn_data = changed model file (used for the first or second hmms in 3-digit model with prob of inter-digit transition as 0.5)
for i in range(0,numclasses):
    with open(classnames[i]+'.hmm','r') as f:
        lines = f.readlines()
        states[i] = int(lines[0].strip('states: '))
        lines = lines[2:]
        nsl = []
        for l in lines:
            if(l != '\n'):
                nsl.append(l)
        orig_data[i] = nsl[:]
        for j in range(1,3):
            nsl[-j] = nsl[-j].strip().split()
            nsl[-j] = list(map(float,nsl[-j]))
            nsl[-j][0] = 0.5
            nsl[-j] = '\t'.join('{:.6f}'.format(e) for e in nsl[-j])+ '\n'
        chn_data[i] = nsl

N_CHMM = sum(states)
M_CHMM = int(input('Enter the number of symbols used: '))

# code to build the 125 possible 3-digit HMMs and create their model files and store their names
# in modelnames list
modelnames = []
for i in range(0,numclasses):
    for j in range(0,numclasses):
        for k in range(0,numclasses):
            modelnames.append(classnames[i]+classnames[j]+classnames[k])
            with open('_'+classnames[i]+classnames[j]+classnames[k]+'.hmm','w') as f:
                f.write('states: '+str(states[i]+states[j]+states[k])+'\n')
                f.write('symbols: '+str(M_CHMM)+'\n\n')
                f.writelines(chn_data[i])
                f.write('\n\n\n')
                f.writelines(chn_data[j])
                f.write('\n\n\n')
                f.writelines(orig_data[k])

# code to build the 25 possible 2-digit HMMs and create their model files and store their names
# in modelnames list
for i in range(0,numclasses):
    for j in range(0,numclasses):
        modelnames.append(classnames[i]+classnames[j])
        with open('_'+classnames[i]+classnames[j]+'.hmm','w') as f:
            f.write('states: '+str(states[i]+states[j])+'\n')
            f.write('symbols: '+str(M_CHMM)+'\n\n')
            f.writelines(chn_data[i])
            f.write('\n\n\n')
            f.writelines(orig_data[j])

# get the codebook for vector quantisation
fname = str(input('Enter the name of the file with the codebook: '))
centroids = extractCodebook(fname)

devDataObsSeq = []
for cname, data in devData:
    devDataObsSeq.append((cname,constructObsSequence(centroids,data)))

scorelists = dict()
for m in modelnames:
    scorelists[m] = testAgainstHMMFile('_'+m+'.hmm',devDataObsSeq)

# truthlist[i], predlist[i] = true class of dev data[i], pred class of dev data[i]
# pred class = class with highest log posterior
truthlist, predlist = [], []
N_DEV = len(devData)
for i in range(0,N_DEV):
    maxscore = np.NINF
    pred = None
    for m in modelnames:
        cscore = scorelists[m][i]
        if(cscore > maxscore):
            pred = m
            maxscore = cscore
    predlist.append(pred)
    truthlist.append(devData[i][0])

# print truth vs prediction for each dev data vector and accuracy of model
correctpred = 0
for i in range(N_DEV):
    print('Truth = '+truthlist[i]+'\t Prediction = '+predlist[i])
    if(predlist[i] == truthlist[i]):
        correctpred += 1

print('Accuracy = {}'.format(correctpred/N_DEV))

# code to extract test data from FOLDER/test/ and build test list
# testData[i] = list of 2-tuples where each tuple D = (trueclass = file name of D, data of D)
testData = []
path = cisodigitsfolder + '/test/'
for fname in os.listdir(path):
    if(fname.__contains__('.mfcc')):
        with open(os.path.join(path,fname),'r') as f:
            testData.append((fname,extractData(f)))

# quantize test data and get the scores of test data against all 150 HMMs
testDataObsSeq = []
for fname, data in testData:
    testDataObsSeq.append((fname,constructObsSequence(centroids,data)))

scorelists = dict()
for m in modelnames:
    scorelists[m] = testAgainstHMMFile('_'+m+'.hmm',testDataObsSeq)

# make predictions assuming equal priors for all models 
predlist = []
N_TEST = len(testData)
for i in range(0,N_TEST):
    maxscore = np.NINF
    pred = None
    for m in modelnames:
        cscore = scorelists[m][i]
        if(cscore > maxscore):
            pred = m
            maxscore = cscore
    predlist.append(pred)

# print blind data predictions
print('The predictions on blind test data are: ')
for i in range(N_TEST):
    print('{} -> {}'.format(testData[i][0],predlist[i]))

os.system('rm _*.hmm')