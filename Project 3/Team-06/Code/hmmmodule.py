# code for helper functions for HMM code
# PRML A3 - Vikram CS19B021, Vedant Saboo CS19B074

from hmmkmeansmodule import KMeans
import numpy as np
import os

# construct a codebook using tData with M symbols and random module seed as s
# returns the kmeans object after fitting tData
def constructCodebook(tData,M,s):
    # extract all data vectors from train data dictionary
    data = []
    for c in tData.keys():
        for d in tData[c]:
            data += d[1].tolist()
    data = np.array(data)
    # init kmeans object with m clusters and max_iter as 10 and fit data on it
    kmeans = KMeans(M,10)
    kmeans = kmeans.fit(data,s)
    return kmeans

# convert the vector series vecSeq to the corr. symbol sequence, i.e, do vector quantisation
def constructObsSequence(kmeans,vecSeq):
    obsSeq = kmeans.predict(vecSeq)
    return obsSeq

# generate a hmm model file with N states, M symbols. Uses 'templates' to generate a temporary file
# temp.seq and generates the hmm model file using train_hmm executable on temp.seq
# the final hmm model file is named as 'fname'
def generateHMMFile(fname, templates, N, M):
    with open('temp.seq','w') as f:
        for _, obsSeq in templates:
            for o in obsSeq:
                f.write(str(o)+' ')
            f.write('\n')
    os.system('./train_hmm temp.seq 1234 '+str(N)+' '+str(M)+' 0.01')
    os.system('mv temp.seq.hmm '+str(fname))
    os.system('rm temp.seq')

# test the dev observation sequences in 'dDataObsSeqs' against the hmm model file 'fname'
# returns the log likelihood that each observation symbol sequence came from model 'fname'
def testAgainstHMMFile(fname, dDataObsSeqs):
    # write all observation symbol sequences in temp.seq in the proper format
    with open('temp.seq','w') as f:
        for _, _, obsSeq in dDataObsSeqs:
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

# save the codebook(centroids' coordinates) for optional part
def savecodebook(kmeans : KMeans,filename):
    with open(filename+'.codebook','w') as f:
        for i in range(0,kmeans.k):
            f.write('\t'.join(str(e) for e in kmeans.centroids[i])+'\n')
