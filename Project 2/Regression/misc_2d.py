import numpy as np

# functions
'''
Phi = [1 x1 x2 x1^2 x1x2 x2^2 ...]

Ms = [1,3,6,10,15,...]

weight vector = [w0 w1 w2 w3...] of dimension M
'''
def phi_of_x1x2(x1, x2, deg):
    y = []
    for d in range(deg+1):
        if(d == 0):
            y.append(1)
        else:
            for t in range(d+1):
                dterm = (x1**(d-t)) * (x2**(t))
                y.append(dterm)
    return np.array(y)
    
def polynomial(x1, x2, *w):
    n = len(w)
    deg = (int(np.sqrt(8*n + 1)) - 3)//2
    try:
        ret = np.dot(w,phi_of_x1x2(x1,x2,deg))
    except:
        ret = 0
        exit()
    return ret

def input(train, dev):
    # io part - get training data
    X1 = []
    X2 = []
    Y = []

    try:
        with open(train, "r") as f:
            data = (f.read()).split('\n')
            for item in data:
                xy = item.split(' ')
                if(len(xy) < 3):
                    break
                X1.append(float(xy[0]))
                X2.append(float(xy[1]))
                Y.append(float(xy[2]))
    except:
        print("Training data not found!")
        exit()
            
    N = X1.__len__()
    X1 = np.array(X1, dtype=float)
    X2 = np.array(X2, dtype=float)
    Y = np.array(Y, dtype=float)

    # io part - get developement/evaluation data
    _X1 = []
    _X2 = []
    _Y = []

    try:
        with open(dev, "r") as f:
            data = (f.read()).split('\n')
            for item in data:
                xy = item.split(' ')
                if(len(xy) < 3):
                    break
                _X1.append(float(xy[0]))
                _X2.append(float(xy[1]))
                _Y.append(float(xy[2]))
    except:
        print("Developement data not found!")
        exit()
            
    _N = _X1.__len__()
    _X1 = np.array(_X1, dtype=float)
    _X2 = np.array(_X2, dtype=float)
    _Y = np.array(_Y, dtype=float)
    
    return X1,X2,Y,N,_X1,_X2,_Y,_N