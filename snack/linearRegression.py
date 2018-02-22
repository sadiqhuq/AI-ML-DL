#!/usr/bin/python

""" Reference: Numerical Recipies (2007)

Fitting Data to a Straight Line 
y(x) = y(x|m,b) = mx + b

"""


import numpy as np
import matplotlib.pyplot as plt

def simple_linear_regression(x, y):
    """
    Find the slope and intercept using mean and covariance
    https://en.wikipedia.org/wiki/Simple_linear_regression#Numerical_example
    """
    m = 0
    b = 0
    c = 0
      

    return m, b

def least_squares_regression(x, y):
    """ General Linear Least Squares """
    m = 0
    b = 0
    c = 0
      
    return m, b, c

# dataset = np.loadtxt ('data.txt', delimiter='\t', dtype=('i4','f4'), skiprows=11)

dataset = np.array([[0,1],[1,0],[2,2],[4,3]])


xt = dataset[:,0]
yt = dataset[:,1]

m_s, b_s = simple_linear_regression (xt, yt)

plt.plot(xt,yt,c='k',marker='o',ls='')

m,b,c = least_squares_regression(xt,yt)

abline =  [m_s * i + b_s for i in xt]
plt.plot(xt,abline,'b', label='simple')

abline =  [m * i + b for i in xt]
plt.plot(xt,abline,'r', label='least square')

plt.legend(loc=0)

plt.show()
