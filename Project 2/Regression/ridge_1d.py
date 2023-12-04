'''
least_sqaures.py
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
2022-02-13 14:50:18 -- file created by vedant
2022-02-17 10:05:29 -- introduced test data evaluation
2022-02-17 12:17:17 -- introduced sampling and mini-batch mode training
2022-02-18 14:43:47 -- bug fixes, introduced bigger and better range for lambda
2022-02-28 16:55:59 -- introduced interactive mode
2022-03-03 00:18:22 -- code modularised

'''

from cmath import inf
import numpy as np
import matplotlib.pyplot as plt
import random
from misc_1d import polynomial
from misc_1d import print_polynomial

# # functions
# def polynomial(x, *w):
#     y = 0
#     for coeff in (list(w)).__reversed__():
#         y = y*x + coeff
#     return y

# def print_polynomial(w):
#     print('f(x) = ',sep='',end='')
#     n = len(w) - 1
#     while(n >= 0):
#         if(n == (len(w) - 1)):
#             print(w[n],'x^',n,sep='',end='')
#         else:
#             if(w[n] >= 0):
#                 print(' + ',sep='',end='')
#             else:
#                 print(' ',sep='',end='')
#             print(w[n],'x^',n,sep='',end='')
#         n -= 1
#     print('')
#     return


    
# # driver code begins here

# # io part - get training data
# X = []
# Y = []

# with open("1d_team_6_train.txt", "r") as f:
#     data = (f.read()).split('\n')
#     for item in data:
#         xy = item.split(' ')
#         if(len(xy) < 2):
#             break
#         X.append(float(xy[0]))
#         Y.append(float(xy[1]))
        
# N = X.__len__()
# X = np.array(X, dtype=float)
# Y = np.array(Y, dtype=float)

# # io part - get developement/evaluation data
# _X = []
# _Y = []

# with open("1d_team_6_dev.txt", "r") as f:
#     data = (f.read()).split('\n')
#     for item in data:
#         xy = item.split(' ')
#         if(len(xy) < 2):
#             break
#         _X.append(float(xy[0]))
#         _Y.append(float(xy[1]))
        
# _N = _X.__len__()
# _X = np.array(_X, dtype=float)
# _Y = np.array(_Y, dtype=float)

def train(input_data, sampling_rate=20, max_degree=7):
    # set a fixed seed for np.random, to force consistency in results
    # for release version, we will remove this restraint
    np.random.seed(6)
    random.seed(6)
    
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
    opt_lambda = 1
    wstarstar = np.array([])

    for M in range(1,max_degree+2):
        lams = np.logspace(-20,20,80)
        for lam in lams:
        
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
                    
                # w* = (Phi^T Phi + Lam I)^-1 Phi^T y    
                
                phit = np.transpose(phi)
                wstar = np.linalg.pinv(phit @ phi + lam*np.identity(M)) @ phit @ sample_y
                wstars.append(np.copy(wstar))
                
            wstar = np.average(wstars, axis=0)
            
            # evaluate on dev data
            
            _phi = np.zeros((_N,M))
            
            for i in range(_N):
                x = _X[i]
                a = 1.0
                for j in range(M - 1):
                    _phi[i][j] = a
                    a *= x
                _phi[i][M-1] = a
            
            # ERMS(D) = SQRT (1/N SUM_{i = 1}^{N} ((t_i - f(x_i,w*)))^2 + Lambda/2 ||w||^2)
            
            Esum = 0
            for i in range(_N):
                # target value
                t = _Y[i]
                # estimated value
                y = wstar @ _phi[i]
                error = np.dot((t - y),(t - y)) + lam/2*np.dot(wstar,wstar)
                Esum += error
            Erms = np.sqrt(Esum / _N)
            
            if(Erms < min_rms):
                min_rms = Erms
                opt_lambda = lam
                opt_M = M
                wstarstar = np.copy(wstar)

    return wstarstar, min_rms, opt_M, opt_lambda

def test_model(results, tfile):
    wstarstar, min_rms, opt_M, opt_lambda = results
    
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
    lam = opt_lambda

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
        error = np.dot((t - y),(t - y)) + lam/2*np.dot(wstarstar,wstarstar)
        Esum += error
    Erms = np.sqrt(Esum / Ntest)

    # print("On test data, Erms =", Erms)

    plt.style.use('seaborn-poster')
    # plot the results
    plt.figure(figsize = (10,8))
    # plt.title('M='+str(M)+',lam='+str(lam)+',sample size='+str(sampling_rate))
    plt.plot(Xtest, Ytest, 'b.')
    plt.plot(Xtest, polynomial(Xtest, *wstarstar) , 'r')
    plt.xlabel('x')
    plt.ylabel('y')
    # plt.savefig('images/1d-ridge-test-'+str(sampling_rate)+'-fig.png')
    plt.show()

    return Erms
