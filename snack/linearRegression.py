#!/usr/bin/python

""" 
Sadiq Huq, Karlsruhe Institute of Technology

Reference: Numerical Recipies (2007)

Fitting Data to a Straight Line 
y(x) = y(x|m,b) = mx + b

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def simple_linear_regression(x, y):
    """
    Find the slope and intercept using
    mean(x), mean(y), covar(x,y), var(x)
    https://en.wikipedia.org/wiki/Simple_linear_regression#Numerical_example
    """
    # norm = len(xt)-1   # NumPy normalizes variance by N-ddof 
    norm = 0
    
    covariance = np.cov(xt,yt,ddof=norm)
    
    m = covariance [0,1] / np.var(xt,ddof=norm)
    b = np.mean(yt) - m * np.mean(xt)

    return m, b

def least_squares_regression(x, y):
    """ General Linear Least Squares """

    m = 0
    b = 0
    c = 0
      
    return m, b, c


def abline (m, b, xt):
    """ Generate straight line to plot """

    return [m * i + b for i in xt]


filename = '../datasets/UCI/iris/iris.data'

df = pd.read_csv(filename,sep=',')

# dataset = np.array([[0,1],[1,0],[2,2],[4,3]])
dataset = df.values

xt = dataset[:,0]
yt = dataset[:,1]

m_s, b_s   = simple_linear_regression (xt, yt)
m_l, b_l,c = least_squares_regression (xt, yt)

plt.plot(xt,yt                  ,c='k', label='sample', marker='o',ls='')
plt.plot(xt,abline(m_s, b_s, xt),c='b', label='simple')
plt.plot(xt,abline(m_l, b_l, xt),c='r', label='least square')

plt.legend(loc=0)

plt.show()
