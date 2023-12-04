'''
least_sqaures.py
----------------------
This is the code for linear regression of
2d data using least squares error.
-------------------------------------------

Author -- Vedant Saboo (CS19B074), K V Vikram (CS19B021)
Team   -- Team 6
Course -- CS 5691 Pattern Recognition and Machine Learning
-----------------------------------------------------------

Log :
2022-02-18 14:54:31 -- file created
2022-02-19 19:29:11 -- introduced 3d plots
2022-02-19 21:03:31 -- introduced custom sampling
2022-03-01 22:57:25 -- introduced iteractive mode
2022-03-03 00:32:24 -- modularised the code

'''

from cmath import inf
import numpy as np
import matplotlib.pyplot as plt
import random
from misc_2d import phi_of_x1x2
from misc_2d import polynomial

def train(input_data, sampling_rate=200, max_degree=6):
    # set a fixed seed for np.random, to force consistency in results
    # for release version, we will remove this restraint
    np.random.seed(6)
    random.seed(6)

    X1,X2,Y,N,_X1,_X2,_Y,_N = input_data

    Data = list(zip(X1, X2, Y))

    random.shuffle(Data)

    samples_zipped = np.array_split(Data, N//sampling_rate)

    samples = []

    for sample in samples_zipped:
        sample_x1, sample_x2, sample_y = zip(*sample)
        sample_x1 = np.array(sample_x1,dtype=float)
        sample_x2 = np.array(sample_x2,dtype=float)
        sample_y = np.array(sample_y,dtype=float)
        ind = np.argsort(sample_x1)
        sample_x1 = sample_x1[ind]
        sample_x2 = sample_x2[ind]
        sample_y = sample_y[ind]
        samples.append([sample_x1, sample_x2,sample_y])

    # training

    # for hyper parameter M

    min_rms = inf
    opt_M = 1
    wstarstar = np.array([])

    for deg in range(0,max_degree+1):
        M = (deg*deg + 3*deg + 2)//2
        
        wstars = []
        
        for sample_count in range(len(samples)):
            # basis matrix phi
            phi = np.zeros((sampling_rate,M))
            sample_x1 = samples[sample_count][0]
            sample_x2 = samples[sample_count][1]
            sample_y = samples[sample_count][2]
            for i in range(sampling_rate):
                phi[i] = phi_of_x1x2(sample_x1[i],sample_x2[i],deg=deg)
                
            wstar = np.linalg.pinv(phi) @ sample_y
            wstars.append(np.copy(wstar))
        
        wstar = np.average(wstars,axis=0)
        
        # evaluate on dev data
        
        _phi = np.zeros((_N,M))
        
        for i in range(_N):
            x1 = _X1[i]
            x2 = _X2[i]
            _phi[i] = phi_of_x1x2(x1,x2,deg=deg)
        
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
    # io part - get test data
    wstarstar, min_rms, opt_M = results
    X1test = []
    X2test = []
    Ytest = []

    try:
        with open(tfile, "r") as f:
            data = (f.read()).split('\n')
            for item in data:
                xy = item.split(' ')
                if(len(xy) < 2):
                    break
                X1test.append(float(xy[0]))
                X2test.append(float(xy[1]))
                Ytest.append(float(xy[2]))
    except IOError:
        print("Test file not present, exiting")
        exit()
            
    Ntest = X1test.__len__()
    X1test = np.array(X1test, dtype=float)
    X2test = np.array(X2test, dtype=float)
    Ytest = np.array(Ytest, dtype=float)

    # evaluate test data

    M = opt_M
    deg = (int(np.sqrt(8*M+1))-3)//2

    _phi = np.zeros((Ntest,M))
        
    for i in range(Ntest):
        x1 = X1test[i]
        x2 = X2test[i]
        _phi[i] = phi_of_x1x2(x1,x2,deg)
        

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

    plt.style.use('seaborn-poster')
    # plot the results
    plots_x1 = []
    plots_x2 = []
    plots_y = []
    l = np.linspace(-1,1,500)
    for i in range(0,500):
        for j in range(0,500):
            plots_x1.append(l[i])
            plots_x2.append(l[j])
            plots_y.append(polynomial(l[i],l[j],*wstarstar))

    fig = plt.figure()
    fig.set_figwidth(10)
    fig.set_figheight(10)
    ax = plt.axes(projection='3d')
    ax.plot3D(plots_x1,plots_x2,plots_y,color='blue', alpha=0.75)
    ax.scatter3D(X1test,X2test,Ytest,color='red')

    plt.show()
    
    return Erms
