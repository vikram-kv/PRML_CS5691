'''
least_squares.py
----------------------
This is the code for linear regression of
1d data using least squares error. It takes
the name of test data file as input and
produces plots and data to show the
regression results.
-------------------------------------------

Author -- Vedant Saboo (CS19B074), K V Vikram (CS19B021)
Team   -- Team 6
Course -- CS 5691 Pattern Recognition and Machine Learning
-----------------------------------------------------------

Log :
2022-02-13 14:50:18 -- file created
2022-02-17 10:05:29 -- introduced test data evaluation
2022-02-17 12:17:17 -- introduced sampling and mini-batch mode training
2022-02-18 14:44:31 -- bug fixes
2022-02-21 18:37:59 -- clean up and fix bugs
2022-02-21 19:49:36 -- fixed the seed for random functions to introdcude consistency in the results
2022-02-28 16:42:39 -- introduced interactive mode
2022-03-02 23:22:09 -- code modularised

'''

from cmath import inf
import numpy as np
import matplotlib.pyplot as plt
import random
from misc_1d import polynomial
from misc_1d import print_polynomial

def train(input_data, sampling_rate=20, max_degree=7):

    # set a fixed seed for np.random, to force consistency in results
    # for release version, we will remove this restraint
    np.random.seed(6)
    random.seed(6)



    '''
    driver code begins here
    it involves
    1. obtaining training data
    2. obtaining developement data
    3. sampling training data with the 'sampling_rate' parameter specified
    --- in this, we divide the training dataset into groups of 'sampling_rate'
    --- sizes randomly, and intending to find the fit polynomial for each of
    --- these samples, we will average all these polynomials columnwise and
    --- output the final polynomial
    4. set initial value(S) for hyper parameter(S), in this case, M.
    5. changing the hyper parameters, we will train the model several times on
    --- the training set, and evaluate in on the developement set
    6. take the hyper parameter with best results on the developement set
    7. obtaining test data
    8. evaluate the fit polynomial on test data, plot results for it
    '''

    X,Y,N,_X,_Y,_N = input_data

    Data = list(zip(X, Y))

    random.shuffle(Data)

    samples_zipped = np.array_split(Data, N//sampling_rate)

    # print(samples_zipped)

    samples = []
    

    for sample in samples_zipped:
        sample_x, sample_y = zip(*sample)
        sample_x = np.array(sample_x,dtype=float)
        sample_y = np.array(sample_y,dtype=float)
        ind = np.argsort(sample_x)
        sample_x = sample_x[ind]
        sample_y = sample_y[ind]
        samples.append([sample_x,sample_y])

    # training

    # for hyper parameter M

    min_rms = inf
    opt_M = 1
    wstarstar = np.array([])

    for M in range(1,max_degree+2):
        
        wstars = []
        
        for sample_count in range(len(samples)):
            # basis matrix phi
            phi = np.zeros((sampling_rate,M))
            sample_x = samples[sample_count][0]
            sample_y = samples[sample_count][1]
            
            for i in range(sampling_rate):
                x = sample_x[i]
                a = 1.0
                for j in range(M - 1):
                    phi[i][j] = a
                    a *= x
                phi[i][M-1] = a
                
            # w* = phi^dagger @ y, where phi^dagger is the Moore-Penrose pseudo-inverse of phi
                
            wstar = np.linalg.pinv(phi) @ sample_y
            wstars.append(np.copy(wstar))
        
        wstar = np.average(wstars,axis=0)
        
        # evaluate on dev data
        
        _phi = np.zeros((_N,M))
        
        for i in range(_N):
            x = _X[i]
            a = 1.0
            for j in range(M - 1):
                _phi[i][j] = a
                a *= x
            _phi[i][M-1] = a
        
        # ERMS(D) = SQRT (1/N SUM_{i = 1}^{N} ((t_i - f(x_i,w*)))^2)
        
        Esum = 0
        for i in range(_N):
            # target value
            t = _Y[i]
            # estimated value
            y = wstar @ _phi[i]
            error = np.dot((t - y),(t - y))
            Esum += error
        Erms = np.sqrt(Esum / _N)
        
        if(Erms < min_rms):
            min_rms = Erms
            opt_M = M
            wstarstar = np.copy(wstar)
        
    return wstarstar, min_rms, opt_M

def test_model(results, tfile):
    wstarstar, min_rms, opt_M = results

    # io part - get test data
    Xtest = []
    Ytest = []

    try:
        with open(tfile, "r") as f:
            data = (f.read()).split('\n')
            for item in data:
                xy = item.split(' ')
                if(len(xy) < 2):
                    break
                Xtest.append(float(xy[0]))
                Ytest.append(float(xy[1]))
    except IOError:
        print("Test file not present, exiting")
        exit()
            
    Ntest = Xtest.__len__()
    Xtest = np.array(Xtest, dtype=float)
    Ytest = np.array(Ytest, dtype=float)

    # evaluate test data

    M = opt_M


    _phi = np.zeros((Ntest,M))
        
    for i in range(Ntest):
        x = Xtest[i]
        a = 1.0
        for j in range(M - 1):
            _phi[i][j] = a
            a *= x
        _phi[i][M-1] = a

    # ERMS(D) = SQRT (1/N SUM_{i = 1}^{N} ((t_i - f(x_i,w*)))^2)

    Esum = 0
    for i in range(Ntest):
        # target value
        t = Ytest[i]
        # estimated value
        
        y = wstarstar @ _phi[i]
        error = np.dot((t - y),(t - y))
        Esum += error
    Erms = np.sqrt(Esum / Ntest)

    # print("On test data, Erms =", Erms)

    plt.style.use('seaborn-poster')
    # plot the results
    plt.figure(figsize = (10,8))
    # plt.title('M='+str(M)+',sample size='+str(sampling_rate))
    plt.plot(Xtest, Ytest, 'b.')
    plt.plot(Xtest, polynomial(Xtest, *wstarstar) , 'r')
    plt.xlabel('x')
    plt.ylabel('y')
    # plt.savefig('images/1d-ls-test-'+str(sampling_rate)+'-fig.png')
    plt.show()
    
    return Erms
