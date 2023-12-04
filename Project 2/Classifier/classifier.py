'''
classifier.py
----------------------
This is the driver code for the construction of 3-class bayesian classifiers for different cases
as mentioned in the problem statement. The code plots the case-comparative ROC and DET curves,
and for a given case(user's choice), it plots PDFs, Decision Boundary Diagram, Equicontour lines of the PDFs
along with the eigenvectors of the class' covariance matrices, and the Confusion matrix.

Note: A feature vector is generally represented as x = (x1,x2) where x1 and x2 are the coordinates of x
      measured along 'x1' and 'x2' axes.
-------------------------------------------
Author -- Vikram - CS19B021, Vedant - CS19B074
Team   -- Team 6
Course -- CS 5691 Pattern Recognition and Machine Learning
'''

import numpy as np
# import plotting and helper functions from our modules
from ConfusionMatrix import ConfusionMatrixPlotter
from PDFPlot import PDFPlotter
from EVAndContourPlot import ConstantDensityAndEigenvectorPlotter
from DCBPlot import DecisionBoundaryPlotter
from ROCDETPlot import ROCDET_curves
from Helpers import mean2D, var1D, cov2D, mean1D
import matplotlib.pyplot as plt
import os

# implementation of the 2D gaussian pdf with input feature vector
# x, mean mu and covariance cov
def gaussianPDF(x, mu, cov):
    det = np.linalg.det(cov)
    dif = x - mu
    cov_inv = np.linalg.inv(cov)
    exp_scalar = (dif.T @ cov_inv @ dif).item(0)
    return np.exp(-exp_scalar)/(2*np.pi*np.sqrt(det))

# In all cases, for each class, mle mean = SUM(x)/n 
# where SUM is over all train vectors from the class and
# n = no of train vectors from the class. Only estimation 
# of covariance will change

'''
    MLE estimator for case 1. c{i}vtrs = train vectors of class i
    returns [mu1,mu2,mu3], [cov1,cov2,cov3] where mu(i), cov(i) are the
    estimated mean and covariance of class i respectively.
'''
def Estimator1(c1vtrs,c2vtrs,c3vtrs):
    vtrs = [c1vtrs,c2vtrs,c3vtrs]
    estclassmeans = [None for i in range(3)]
    estclasscovs = [None for i in range(3)]

    # common cov of every class is SUM[(x-mu).(x-mu)t]/N where SUM is over all train 
    # vectors, mu is the estimated mean of class of x, N = total number of train vectors 
    covarsum = np.zeros((2,2),dtype=np.float64)
    N = 0
    for i in range(0,3):
        estclassmeans[i] = mean2D(vtrs[i])
        mean = estclassmeans[i]
        N += vtrs[i].shape[0]
        for v in vtrs[i]:
            diff = v - mean
            covarsum += np.outer(diff,diff)

    commoncov = covarsum/N
    return estclassmeans, [commoncov for i in range(0,3)]

'''
    MLE estimator for case 2. c{i}vtrs = train vectors of class i
    returns [mu1,mu2,mu3], [cov1,cov2,cov3] where mu(i), cov(i) are the
    estimated mean and covariance of class i respectively.
'''
def Estimator2(c1vtrs,c2vtrs,c3vtrs):
    vtrs = [c1vtrs,c2vtrs,c3vtrs]
    estclassmeans = [None for i in range(3)]
    estclasscovs = [None for i in range(3)]

    # for every class c, cov(c) = SUM[(x-mu).(x-mu)t]/N where SUM is over all train vectors 
    # from class c, mu is the estimated mean of class c, N = total number of train vectors from class c
    for i in range(0,3):
        estclassmeans[i] = mean2D(vtrs[i])
        estclasscovs[i] = cov2D(vtrs[i])
    
    return estclassmeans, estclasscovs

'''
    MLE estimator for case 3. c{i}vtrs = train vectors of class i
    returns [mu1,mu2,mu3], [cov1,cov2,cov3] where mu(i), cov(i) are the
    estimated mean and covariance of class i respectively.
'''
def Estimator3(c1vtrs,c2vtrs,c3vtrs):
    vtrs = [c1vtrs,c2vtrs,c3vtrs]
    estclassmeans = [None for i in range(3)]
    estclasscovs = [None for i in range(3)]

    # here, all classes have the same cov of the form var.I,
    # var = SUM(norm(v-mu)^2)/(2*N), where SUM is over all train 
    # vectors v, mu is the estimated mean of class of v, 
    # N = total number of train vectors 
    varsum = 0.0
    N = 0
    for i in range(0,3):
        estclassmeans[i] = mean2D(vtrs[i])
        mean_x1, mean_x2 = estclassmeans[i][0], estclassmeans[i][1]
        N += vtrs[i].shape[0]
        for v in vtrs[i]:
            varsum += (v[0]-mean_x1) ** 2 + (v[1]-mean_x2) ** 2
    
    var = varsum/(2*N)
    scalar_cov = np.asarray([[var, 0], [0, var]], dtype=np.float64)
    return estclassmeans, [scalar_cov for i in range(0,3)]

'''
    MLE estimator for case 4. c{i}vtrs = train vectors of class i
    returns [mu1,mu2,mu3], [cov1,cov2,cov3] where mu(i), cov(i) are the
    estimated mean and covariance of class i respectively.
'''
def Estimator4(c1vtrs,c2vtrs,c3vtrs):
    vtrs = [c1vtrs,c2vtrs,c3vtrs]
    estclassmeans = [None for i in range(3)]
    estclasscovs = [None for i in range(3)]

    # here, all classes have the same diagonal cov matrix C = diag(var1,var2), then
    # var1 = SUM((x1(v)-x1(mu))^2)/N, var2 = SUM((x2(v)-x2(mu))^2)/N where SUM is over 
    # all train vectors v, mu is the estimated mean of class of v, x{i}(v) = component of 
    # vector v along axis x{i}, N = total number of train vectors 
    varsum_x1 = 0.0
    varsum_x2 = 0.0
    N = 0
    for i in range(0,3):
        estclassmeans[i] = mean2D(vtrs[i])
        mean_x1, mean_x2 = estclassmeans[i][0], estclassmeans[i][1]
        N += vtrs[i].shape[0]
        for v in vtrs[i]:
            varsum_x1 += (v[0]-mean_x1) ** 2
            varsum_x2 += (v[1]-mean_x2) ** 2
    
    var_x1 = varsum_x1/N
    var_x2 = varsum_x2/N
    commoncov = np.asarray([[var_x1, 0], [0, var_x2]], dtype=np.float64)
    return estclassmeans, [commoncov for i in range(0,3)]

'''
    MLE estimator for case 5. c{i}vtrs = train vectors of class i
    returns [mu1,mu2,mu3], [cov1,cov2,cov3] where mu(i), cov(i) are the
    estimated mean and covariance of class i respectively.
'''
def Estimator5(c1vtrs,c2vtrs,c3vtrs):
    vtrs = [c1vtrs,c2vtrs,c3vtrs]
    estclassmeans = [None for i in range(3)]
    estclasscovs = [None for i in range(3)]

    # here, if a class c has the cov matrix C = diag(var1,var2), then var1 = SUM((x1(v)-x1(mu))^2)/N, 
    # var2 = SUM((x2(v)-x2(mu))^2)/N where SUM is over all train vectors v from class c, mu is the 
    # estimated mean of class c, x{i}(v) = component of vector v along axis x{i}, 
    # N = total number of train vectors from class c
    for i in range(3):
        estclassmeans[i] = mean2D(vtrs[i])
        var_x1, var_x2 = var1D(vtrs[i][:,0]), var1D(vtrs[i][:,1])
        estclasscovs[i] = np.asarray([[var_x1, 0], [0, var_x2]], dtype=np.float64)
    
    return estclassmeans, estclasscovs

# fetch train data, build numpy arrays from them
# class{i}_vtrs = train feature vectors from class i
# train_vtrs = train feature vectors from all classes
class1_vtrs = []
class2_vtrs = []
class3_vtrs = []
train_vtrs = []
path = str(input('Enter file name for training data with rel. path:'))
with open(path,'r') as f:
    data = f.readlines()
    for l in data:
        l = l.split(',')
        _class = int(l[2])
        if(_class == 1):
            class1_vtrs.append([l[0],l[1]])
        elif(_class == 2):
            class2_vtrs.append([l[0],l[1]])
        else:
            class3_vtrs.append([l[0],l[1]])
        train_vtrs.append([l[0],l[1]])
class1_vtrs = np.asarray(class1_vtrs,dtype=np.float64)
class2_vtrs = np.asarray(class2_vtrs,dtype=np.float64)
class3_vtrs = np.asarray(class3_vtrs,dtype=np.float64)
train_vtrs = np.asarray(train_vtrs,dtype=np.float64)

# compute priors for each class
# prior of class c = (num of train vectors from c)/(total number of train vectors)
prior = np.zeros(shape=(3,),dtype=np.float64)
n1, n2, n3, n = class1_vtrs.shape[0], class2_vtrs.shape[0], class3_vtrs.shape[0], train_vtrs.shape[0]
prior[0], prior[1], prior[2] = n1/n, n2/n, n3/n

# fetch development data into dev_vtrs array
# truth[i] = true class of dev_vtrs[i]
truth = []
dev_vtrs = []
path = str(input('Enter file name for development data with rel. path:'))
with open(path,'r') as f:
    data = f.readlines()
    for l in data:
        l = l.split(',')
        truth.append(int(l[2]))
        dev_vtrs.append([float(l[0]),float(l[1])])
dev_vtrs = np.asarray(dev_vtrs,dtype=np.float64)


# At this stage, we can do some type of normalisation to ensure that one feature dimension does not contribute more than
# the other. We experimented with max-min normalization and Z score normalization. However, there was no appreciable
# improvement in the classification with both those types. So, this is our final code. However, we have given the 2 normalization
# codes at the end of this file. Any one of them can be placed here to perform normalization also. No additional changes are required.


# plot range calculation 
# we need a square region aligned along the x1 and x2 axes in order
# to get correct contour curves using a meshgrid.
# here, [_range[0][0], _range[0][1] ] is sampling interval for x1
# and [_range[1][0], _range[1][1] ] is sampling interval for x2
# sampling intervals are for making pdf plot and decision boundary diagrams
_range = np.empty((2,2))
for i in [0,1]:
    _range[i][0] = np.amin([np.amin(train_vtrs[:,i]), np.amin(dev_vtrs[:,i])])
    _range[i][0] = _range[i][0] - 0.05*np.abs(_range[i][0])
    _range[i][1] = np.amax([np.amax(train_vtrs[:,i]), np.amax(dev_vtrs[:,i])])
    _range[i][1] = _range[i][1] + 0.05*np.abs(_range[i][1])

side = np.amax([_range[0][1] - _range[0][0], _range[1][1] - _range[1][0]])
mid = [(_range[0][1] + _range[0][0])/2, (_range[1][1] + _range[1][0])/2]
_range[0][0] = mid[0] - side/2
_range[0][1] = mid[0] + side/2
_range[1][0] = mid[1] - side/2
_range[1][1] = mid[1] + side/2

# parameter estimation for all 5 cases * 3 classes
# estimator list
Estimators = [Estimator1, Estimator2, Estimator3, Estimator4, Estimator5]
# class case mean matrix = clcmum
clcmum = [[ ], [ ], [ ]]
# class case cov matrix = clccovm
clccovm = [[ ], [ ], [ ]]
# loop to estimate mus, covs for each case and build clcmum, clccovm
for i in range(0,5):
    mus, covs = (Estimators[i])(class1_vtrs,class2_vtrs,class3_vtrs)
    clcmum[0].append(mus[0]), clcmum[1].append(mus[1]), clcmum[2].append(mus[2])
    clccovm[0].append(covs[0]), clccovm[1].append(covs[1]), clccovm[2].append(covs[2])

# get the case number _cse from the user for plotting other curves
folder_name = str(input("Enter a name nm (All plots will be put in a new folder (nm)_plots):"))
os.system('rm -r '+folder_name+'_plots')
os.system('mkdir '+folder_name+'_plots')

for _cse in range(1,6):
    i = _cse-1
    c1_mu, c2_mu, c3_mu = clcmum[0][i], clcmum[1][i], clcmum[2][i]
    c1_cov, c2_cov, c3_cov = clccovm[0][i], clccovm[1][i], clccovm[2][i]

    # mus, covs = list of means, covs for case _cse
    mus = [c1_mu, c2_mu, c3_mu]
    covs = [c1_cov, c2_cov, c3_cov]

    # building classifier as argmax of posterior
    post1 = lambda x: prior[0] * gaussianPDF(x,c1_mu,c1_cov)
    post2 = lambda x: prior[1] * gaussianPDF(x,c2_mu,c2_cov)
    post3 = lambda x: prior[2] * gaussianPDF(x,c3_mu,c3_cov)
    classifier = lambda x : np.argmax([post1(x),post2(x),post3(x)]) + 1

    # making predictions for development data using classifier
    pred = []
    for v in dev_vtrs:
        pred.append(classifier(v))

    # plotting section for pdfs, dcbs, contours+eigenvectors, confusion matrix for the case _cse
    PDFPlotter(_range,mus,covs)
    # Here, the orientation in the saved figure might not be proper. We have manually oriented for each case
    # to get the plots shown in the reports
    plt.savefig(folder_name+'_plots/PDFs_Case'+str(_cse)+'.png')
    plt.show()
    DecisionBoundaryPlotter(_range,mus,covs,dev_vtrs,truth,classifier)
    plt.savefig(folder_name+'_plots/DecisionBoundary_Case'+str(_cse)+'.png')
    plt.show()
    ConstantDensityAndEigenvectorPlotter(covs,mus,_range)
    plt.savefig(folder_name+'_plots/EigenvectorContour_Case'+str(_cse)+'.png')
    plt.show()
    ConfusionMatrixPlotter(truth,pred)
    plt.savefig(folder_name+'_plots/ConfusionMatrix_Case'+str(_cse)+'.png')
    plt.show()

# ROC/DET Plotting Section
# ccsm - class case scorer matrix
# ccsm[i][j] = scorer of class i+1 for case j+1.
# so ccsm is a 3 * 5 matrix of functions
ccsm = [[ ], [ ], [ ]]
for i in range(0,5):
    ccsm[0].append(lambda x,i=i: prior[0] * gaussianPDF(x,clcmum[0][i],clccovm[0][i]))
    ccsm[1].append(lambda x,i=i: prior[1] * gaussianPDF(x,clcmum[1][i],clccovm[1][i]))
    ccsm[2].append(lambda x,i=i: prior[2] * gaussianPDF(x,clcmum[2][i],clccovm[2][i]))
ROCDET_curves(dev_vtrs,truth,ccsm)
plt.savefig(folder_name+'_plots/ROCDETPlot.png')


# Max-Min normalization
'''
x1_list = []
x2_list = []

for t in train_vtrs:
    x1_list += [t[0]]
    x2_list += [t[1]]

for d in dev_vtrs:
    x1_list += [d[0]]
    x2_list += [d[1]]

x1_list = np.asarray(x1_list,dtype=np.float64)
x2_list = np.asarray(x2_list,dtype=np.float64)
min_x1, max_x1 = np.amin(x1_list), np.amax(x1_list)
min_x2, max_x2 = np.amin(x2_list), np.amax(x2_list)

for i in range(n):
    train_vtrs[i][0] = 1000*(train_vtrs[i][0] - min_x1)/(max_x1-min_x1)
    train_vtrs[i][1] = 1000*(train_vtrs[i][1] - min_x2)/(max_x2-min_x2)

for i in range(n1):
    class1_vtrs[i][0] = 1000*(class1_vtrs[i][0] - min_x1)/(max_x1-min_x1)
    class1_vtrs[i][1] = 1000*(class1_vtrs[i][1] - min_x2)/(max_x2-min_x2)

for i in range(n2):
    class2_vtrs[i][0] = 1000*(class2_vtrs[i][0] - min_x1)/(max_x1-min_x1)
    class2_vtrs[i][1] = 1000*(class2_vtrs[i][1] - min_x2)/(max_x2-min_x2)

for i in range(n3):
    class3_vtrs[i][0] = 1000*(class3_vtrs[i][0] - min_x1)/(max_x1-min_x1)
    class3_vtrs[i][1] = 1000*(class3_vtrs[i][1] - min_x2)/(max_x2-min_x2)

for i in range(dev_vtrs.shape[0]):
    dev_vtrs[i][0] = 1000*(dev_vtrs[i][0] - min_x1)/(max_x1-min_x1)
    dev_vtrs[i][1] = 1000*(dev_vtrs[i][1] - min_x2)/(max_x2-min_x2)
'''


# Z score normalization
'''
x1_list = []
x2_list = []

for t in train_vtrs:
    x1_list += [t[0]]
    x2_list += [t[1]]

for d in dev_vtrs:
    x1_list += [d[0]]
    x2_list += [d[1]]

x1_list = np.asarray(x1_list,dtype=np.float64)
x2_list = np.asarray(x2_list,dtype=np.float64)
normmean_x1, normvar_x1 = mean1D(x1_list), var1D(x1_list)
normmean_x2, normvar_x2 = mean1D(x2_list), var1D(x2_list)

for i in range(n):
    train_vtrs[i][0] = 1000*(train_vtrs[i][0] - normmean_x1)/(normvar_x1)
    train_vtrs[i][1] = 1000*(train_vtrs[i][1] - normmean_x2)/(normvar_x2)

for i in range(n1):
    class1_vtrs[i][0] = 1000*(class1_vtrs[i][0] - normmean_x1)/(normvar_x1)
    class1_vtrs[i][1] = 1000*(class1_vtrs[i][1] - normmean_x2)/(normvar_x2)

for i in range(n2):
    class2_vtrs[i][0] = 1000*(class2_vtrs[i][0] - normmean_x1)/(normvar_x1)
    class2_vtrs[i][1] = 1000*(class2_vtrs[i][1] - normmean_x2)/(normvar_x2)

for i in range(n3):
    class3_vtrs[i][0] = 1000*(class3_vtrs[i][0] - normmean_x1)/(normvar_x1)
    class3_vtrs[i][1] = 1000*(class3_vtrs[i][1] - normmean_x2)/(normvar_x2)

for i in range(dev_vtrs.shape[0]):
    dev_vtrs[i][0] = 1000*(dev_vtrs[i][0] - normmean_x1)/(normvar_x1)
    dev_vtrs[i][1] = 1000*(dev_vtrs[i][1] - normmean_x2)/(normvar_x2)
'''
