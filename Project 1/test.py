from configparser import Interpolation
import numpy as np
import matplotlib.image as img
import matplotlib.pyplot as plt
import os
import shutil
import svd_custom as svd_provider

sampling_rate = 1

'''
saveimg
-------
parameters :
mat     -   2d matrix (numpy.array)
fname   -   name of the input image
k       -   number of eigen values used (k)
returns    :
purpose    :
saves the reconstructed image in the appropriate format (svd)
'''
def saveimg(mat, fname, k):
    plt.figure(1)
    plt.ioff()
    plt.grid(False)
    plt.gray()
    plt.imsave('solution/'+os.path.basename(fname).split('.',1)[0]+'/recon@'+ str(k) +'_svd.jpg', mat)
    return

'''
saverrorimg
-------
parameters :
mat     -   2d error matrix (numpy.array)
fname   -   name of the input image
k       -   number of eigen values used (k)
returns    :
purpose    :
saves the error image in the appropriate format (svd)
'''
def saveerrorimg(mat, fname, k):
    plt.figure(1)
    plt.ioff()
    plt.grid(False)
    plt.gray()
    plt.imsave('solution/'+os.path.basename(fname).split('.',1)[0]+'/error@'+ str(k) +'_svd.jpg', mat)
    return

'''
plot
-------
parameters :
K       -   values of K (x axis)
Norms   -   values of frobenius norms (measure of error) (y axis)
fname   -   name of the input image
returns    :
purpose    :
plots and saves the graph in the appropriate format (svd)
'''
def plot(K, Norms, fname):
    plt.plot(K, Norms)
    plt.title('Plot for SVD')
    plt.xlabel('K')
    plt.ylabel('Norms')
    plt.ioff()
    plt.savefig('solution/'+os.path.basename(fname).split('.',1)[0]+'/plot_svd.jpg')
    plt.close()

'''
saveimg_evd
-------
parameters :
mat     -   2d matrix (numpy.array)
fname   -   name of the input image
k       -   number of eigen values used (k)
returns    :
purpose    :
saves the reconstructed image in the appropriate format (evd)
'''
def saveimg_evd(mat, fname, k):
    plt.figure(1)
    plt.ioff()
    plt.grid(False)
    plt.gray()
    plt.imsave('solution/'+os.path.basename(fname).split('.',1)[0]+'/recon@'+ str(k) +'_evd.jpg', mat)
    return

'''
saverrorimg_evd
-------
parameters :
mat     -   2d error matrix (numpy.array)
fname   -   name of the input image
k       -   number of eigen values used (k)
returns    :
purpose    :
saves the error image in the appropriate format (evd)
'''
def saveerrorimg_evd(mat, fname, k):
    plt.figure(1)
    plt.ioff()
    plt.grid(False)
    plt.gray()
    plt.imsave('solution/'+os.path.basename(fname).split('.',1)[0]+'/error@'+ str(k) +'_evd.jpg', mat)
    return

'''
plot
-------
parameters :
K       -   values of K (x axis)
Norms   -   values of frobenius norms (measure of error) (y axis)
fname   -   name of the input image
returns    :
purpose    :
plots and saves the graph in the appropriate format (evd)
'''
def plot_evd(K, Norms, fname):
    plt.plot(K, Norms)
    plt.title('Plot for EVD')
    plt.xlabel('K')
    plt.ylabel('Norms')
    plt.ioff()
    plt.savefig('solution/'+os.path.basename(fname).split('.',1)[0]+'/plot_evd.jpg')
    plt.close()

'''
fnorm
------
frobenius norm of the difference of two matrices of same shape
parameters :
A       -   first matrix
B       -   second matrix
returns    :
sqrt(sum_over_ij((Aij-Bij)^2))
side effects : none
'''
def fnorm(A, B):
    diff = A - B
    return np.linalg.norm(diff,'fro')

'''
complex_sorter
--------------
rank function used to sorting complex numbers
negative of the magnitude, 
    implying that sorter will sort them descending in magnitudes
'''
def complex_sorter(x):
    return -np.abs(x)

f = '72.jpg'
try:
    os.makedirs('solution/'+os.path.basename(f).split('.',1)[0])
except:
    shutil.rmtree('solution/'+os.path.basename(f).split('.',1)[0])
    os.makedirs('solution/'+os.path.basename(f).split('.',1)[0])
    # form solution directory structure
mat = np.array(img.imread('dataset/'+f), dtype=np.double)       # read the grey scale square image
rows, cols = mat.shape                                          # record rows and cols

# svd_provider provides with the user-implemented svd solution
U,D,Vt = svd_provider.svd(mat)
evals_pred = complex_sorter(D)
ind = np.argsort(evals_pred)
titer = rows // sampling_rate
K = []
Norms = []
D_ = [x for x in D]
c_ind = len(D_)
while (titer > 0):
    D1 = []
    for i in range(rows):
        row = []
        for j in range(cols):
            if(i is j):
                row.append(D_[i])
            else:
                row.append(float(0))
        D1.append(row)
    k = c_ind
    K.insert(0, k)
    mat_ = np.abs(U @ D1 @ Vt)
    saveimg(mat_, f, k)
    saveerrorimg(np.abs(mat - mat_), f, k)
    norm = fnorm(mat, mat_)
    Norms.insert(0, norm)
    titer -= 1
    for _ in range (sampling_rate):
        c_ind -= 1
        D_[ind[c_ind]] = float(0)
plot(K, Norms, f)

# evd
evals, evectors = np.linalg.eig(mat)
evals_pred = complex_sorter(evals)
ind = np.argsort(evals_pred)
titer = rows // sampling_rate
K = []
Norms = []
D1 = np.array([x for x in evals])
c_ind = len(D1)
while (titer > 0):
    k = c_ind
    K.insert(0, k)
    mat_ = np.abs(evectors @ np.diag(D1) @ np.linalg.inv(evectors))
    saveimg_evd(mat_, f, k)
    saveerrorimg_evd(np.abs(mat - mat_), f, k)
    norm = fnorm(mat, mat_)
    Norms.insert(0, norm)
    titer -= 1
    for _ in range (sampling_rate):
        c_ind -= 1
        D1[ind[c_ind]] = complex(0)
plot_evd(K, Norms, f)
    