# module that implements DTW algorithm (with and without windows).
# PRML A3 - Vikram CS19B021, Vedant Saboo CS19B074
# requires numba library to speed up DTW algos

from numba import jit
import numpy as np

# dtw error calculator without windows
@jit(nopython=True)
def dtw(ref,test):
    # N, M = number of feature vectors in reference series and test series resp.
    N = ref.shape[0]
    M = test.shape[0]

    # init DTW matrix
    costarray = np.full((M+1,N+1),np.PINF,dtype=np.double)
    costarray[0,0] = 0.0

    # nested loops to implement DTW DP algo
    for i in range(1,M+1):
        for j in range(1,N+1):
            # find cost(error) in matching ith symbol of test against jth symbol of ref
            cost = np.linalg.norm(test[i-1] - ref[j-1])
            # total cost of matching test[1..i] and ref[1..j]
            costarray[i,j] = cost + min([costarray[i-1,j],costarray[i,j-1],costarray[i-1,j-1]])
    
    return costarray[M,N]   # total error in matching test[1..m] and ref[1..n]

# dtw error calculator with a window of size wsize
@jit(nopython=True)
def dtw_window(ref,test,wsize):
    # N, M = number of feature vectors in reference series and test series resp.
    N = ref.shape[0]
    M = test.shape[0]

    w = max(wsize,np.abs(N-M))          # to ensure all symbols get matched, we make the window size bigger if needed

    costarray = np.full((M+1,N+1),np.PINF,dtype=np.double)
    costarray[0,0] = 0.0

    # init all entries within the window as 0
    for i in range(1,M+1):
        for j in range(max(1,i-w),min(N,i+w)+1):
            costarray[i,j] = 0.0

    for i in range(1,M+1):
        # for matching symbol i in test we only look from i-w to i+w with sanity checks
        for j in range(max(1,i-w),min(N,i+w)+1):
            # find cost(error) in matching ith symbol of test against jth symbol of ref
            cost = np.linalg.norm(test[i-1] - ref[j-1])
            # total cost of matching test[1..i] and ref[1..j]
            costarray[i,j] = cost + min([costarray[i-1,j],costarray[i,j-1],costarray[i-1,j-1]])
    
    return costarray[M,N] # total error in matching test[1..m] and ref[1..n]

# function to match test series and all template series to find resp. dtw errors. It then sorts the dtw errors and averages lowest TOP_N
# scores to get the score of test for the template data's class.
# if ch is YES, we use a window of size wsize during DTW. O.W, we perform normal DTW
def findScore(templates,test,TOP_N,choice,wsize):
    if(choice == 'YES'):
        distlist = np.array([dtw_window(ref,test,wsize) for _, ref in templates],dtype=np.double)
    else:
        distlist = np.array([dtw(ref,test) for _, ref in templates],dtype=np.double)
    distlist = np.sort(distlist)
    distlist = distlist[:TOP_N]
    return np.average(distlist)