import numpy as np

# functions
'''
polynomial (x, *w)
------------------
find the value of the w^T @ phi(x) polynomial at value x 
'''
def polynomial(x, *w):
    y = 0
    for coeff in (list(w)).__reversed__():
        y = y*x + coeff
    return y

'''
print_polynomial (w)
--------------------
prints formally the polynomial with weight vector w
'''
def print_polynomial(w):
    print('f(x) = ',sep='',end='')
    n = len(w) - 1
    while(n >= 0):
        if(n == (len(w) - 1)):
            print(w[n],'x^',n,sep='',end='')
        else:
            if(w[n] >= 0):
                print(' + ',sep='',end='')
            else:
                print(' ',sep='',end='')
            print(w[n],'x^',n,sep='',end='')
        n -= 1
    print('')
    return
    
    
def input(training_file,dev_file):
    # io part - get training data
    X = []
    Y = []

    try:
        with open(training_file, "r") as f:
            data = (f.read()).split('\n')
            for item in data:
                xy = item.split(' ')
                if(len(xy) < 2):
                    break
                X.append(float(xy[0]))
                Y.append(float(xy[1]))
    except:
        print("Training data not found!")
        exit()
            
    N = X.__len__()
    X = np.array(X, dtype=float)
    Y = np.array(Y, dtype=float)

    # io part - get developement/evaluation data
    _X = []
    _Y = []
    try:
        with open(dev_file, "r") as f:
            data = (f.read()).split('\n')
            for item in data:
                xy = item.split(' ')
                if(len(xy) < 2):
                    break
                _X.append(float(xy[0]))
                _Y.append(float(xy[1]))
    except:
        print("Developement data not found!")
        exit()
            
    _N = _X.__len__()
    _X = np.array(_X, dtype=float)
    _Y = np.array(_Y, dtype=float)
    
    return X,Y,N,_X,_Y,_N
