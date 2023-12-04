from numpy import ndarray
from inputmodule import inputHCD,inputIMG,inputISD,inputSYN
from scipy.special import expit
import numpy as np
from tqdm.notebook import tqdm_notebook as pbar
from sklearn.preprocessing import OneHotEncoder 

class LogisticRegression:
    def __init__(self, mode='binary'):
        self.mode = mode
        if(mode=='binary'):
            # do something
            self.num_classes = 2
            self.input = inputSYN
            pass
        elif(mode=='multi_class'):
            # do some other thing
            self.num_classes = 5
            self.input = inputIMG
            pass
        else:
            pass
        pass
    
    def dot_sigmoid(x:np.ndarray,y:np.ndarray):
        try:
            z = np.dot(x,y)
        except:
            print(x)
            print(y)
            x = np.array(x).ravel()
            z = np.dot(x,y)
        return expit(z)
    
    def neg_log_likelihood(y: float, y_pred: float):
        return -((y * np.log(y_pred)))

    def error(ys: np.ndarray, ys_pred: np.ndarray):
        assert len(ys) == len(ys_pred)
        num_items: int = len(ys)
        sum_nll: float = np.sum([LogisticRegression.neg_log_likelihood(y, y_pred) for y, y_pred in zip(ys, ys_pred)])
        return (1 / num_items) * sum_nll
    
    def fit(self,X,Y):
        if(self.mode=='binary'):
            return self.fit_bin(X,Y)
        else:
            return self.fit_multi(X,Y)
    
    def fit_bin(self,X,Y):
        # prepend 1 to x to get z
        Z = []
        for x in X:
            z = [x1 for x1 in x]
            z.insert(0,1)
            Z.append(z)
        Z = np.array(Z)
        beta = np.zeros(Z.shape[1])

        epochs: int = 10000
        learning_rate: float = 0.0001

        for epoch in pbar(range(epochs)):
            # Calculate the "predictions" (squishified dot product of `beta` and `x`) based on our current `beta` vector
            ys_pred = np.array([LogisticRegression.dot_sigmoid(beta, x) for x in Z])
            # Calculate and print the error
            if epoch % 10 == True:
                loss: float = LogisticRegression.error(Y, ys_pred)
                print(f'Epoch {epoch} --> loss: {loss}')
                if(loss < 0.08):
                    break
                

            # Calculate the gradient
            grad = [0. for _ in range(len(beta))]
            for x, y in zip(Z, Y):
                try:
                    err = LogisticRegression.dot_sigmoid(beta, x) - y
                except:
                    print(type(beta))
                    print(type(x))
                    print(type(y))
                for i, x_i in enumerate(x):
                    grad[i] += (err * x_i)
            grad = [1 / len(x) * g_i for g_i in grad]

            # Take a small step in the direction of greatest decrease
            beta = np.array([b + (gb * -learning_rate) for b, gb in zip(beta, grad)]).ravel()
            # print(f'Epoch {epoch} beta: {beta}')

        self.beta = np.copy(beta)
        return beta
    
    def fit_multi(self,X,Y):
        # prepend 1 to x to get z
        Z = []
        for x in X:
            z = [x1 for x1 in x]
            z.insert(0,1)
            Z.append(z)
        Z = np.array(Z)
        beta = np.zeros((self.num_classes,Z.shape[1]))
        
        encoder = OneHotEncoder(sparse=False)
        Y = encoder.fit_transform(np.array(Y).reshape(-1,1))

        epochs: int = 10000
        learning_rate: float = 0.0001

        for c_i in range(self.num_classes):
            for epoch in pbar(range(epochs)):
                # Calculate the "predictions" (squishified dot product of `beta` and `x`) based on our current `beta` vector
                ys_pred = np.array([LogisticRegression.dot_sigmoid(beta[c_i], x) for x in Z])
                # Calculate and print the error
                if epoch % 10 == True:
                    loss: float = LogisticRegression.error(Y[:,c_i], ys_pred)
                    print(f'Epoch {epoch} --> loss: {loss}')
                    if(loss < 0.08):
                        break
                    

                # Calculate the gradient
                grad = [0. for _ in range(len(beta[c_i]))]
                for x, y in zip(Z, Y[:,c_i]):
                    try:
                        err = LogisticRegression.dot_sigmoid(beta[c_i], x) - y
                    except:
                        pass
                    for i, x_i in enumerate(x):
                        grad[i] += (err * x_i)
                grad = [1 / len(x) * g_i for g_i in grad]

                # Take a small step in the direction of greatest decrease
                try:
                    beta[c_i] = np.array([b + (gb * -learning_rate) for b, gb in zip(beta[c_i], grad)]).ravel()
                except:
                    print(beta.shape, i, grad)
                # print(f'Epoch {epoch} beta: {beta}')

        self.beta = np.copy(beta)
        return beta
    
    def predict_proba(self, dev_data:ndarray):
        Z = []
        for x in dev_data:
            z = [x1 for x1 in x]
            z.insert(0,1)
            Z.append(z)
        Z = np.array(Z)
        if(self.mode=='binary'):
            ys_pred = np.array([LogisticRegression.dot_sigmoid(self.beta, x) for x in Z])
        else:
            ys_pred = np.array([[LogisticRegression.dot_sigmoid(self.beta[i], x) for x in Z] for i in range(self.num_classes)]).T
        return ys_pred
