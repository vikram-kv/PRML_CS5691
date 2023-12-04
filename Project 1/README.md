# CS5691 Pattern Recognition and Machine Learning 

**Assignment 1**

**Vedant Saboo**

**CS19B074**

## Submission details

1. dataset -- dataset folder provided as a part of problem statement
2. solution -- after iteration of the algorithm, resulting reconstructions and error images will be generated here
3. main.py -- main script for **all** of the dataset
4. test.py -- specific script for detailed (using high sampling) studying for '72.jpg' file (the specific case assigned to me)
5. svd_custom -- svd algorithm designed and implemented by me

## How to use

In this directory, run
    
    python3 main.py

to run all the cases. please wait for a while to get the results in the solution.
Or, run 

    python3 test.py
to run the specific case '72.jpg'

The solution/x folder, for x.jpg will contain 

1. _recon@k_svd.jpg_ files, reconstructions using svd for k maximum eigenvalues
2. _recon@k_evd.jpg_ files, reconstructions using evd for k maximum eigenvalues
3. _error@k_svd.jpg_ files, error image using svd for k maximum eigenvalues
4. _error@k_evd.jpg_ files, error image using svd for k maximum eigenvalues
5. _plot_evd.jpg_, graph of norm vs k for evd 
6. _plot_svd.jpg_, graph of norm vs k for svd 