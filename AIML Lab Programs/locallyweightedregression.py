import math
import numpy as np
from scipy import linalg

def lowess(x, y, f=0.25, iter=3):
    n = len(x)
    r = int(math.ceil(f*n))
    h = [np.sort(np.abs(x-x[i]))[r] for i in range(n)]

    w = np.clip(np.abs((x[:,None] - x[None,:])/h), 0.0, 1.0)
    w = (1-w**3)**3

    yest = np.zeros(n)
    delta = np.ones(n)


    for iteration in range(iter):
        for i in range(n):
            weights = delta*w[:, i]

            b = np.array([np.sum(weights*y), np.sum(weights*y*x)])

            A = np.array([ [ np.sum(weights), np.sum(weights*x)], [ np.sum(weights*x), np.sum(weights*x*x)]])

            beta = linalg.solve(A,b)

            yest[i] = beta[0] +beta[1]*x[i]
    
    return yest 


if __name__== '__main__':
    n=100
    x = np.linspace(0, 2*math.pi, n)
    print("Value of x------------------------ ")
    print(x)
    
    y = np.sin(x) + 0.3*np.random.randn(n)
    print("Value of y------------------------ ")
    print(y)

    f = 0.25
    yest = lowess(x,y,f = f, iter=3)

    import pylab as pl
    pl.clf()
    pl.plot(x,y,label="y noisy")
    pl.plot(x,yest, label='y pred')
    pl.legend()
    pl.show()