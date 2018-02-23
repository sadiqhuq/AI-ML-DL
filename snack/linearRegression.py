#!/usr/bin/python3

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


def abline (xt, m, b):
    """ Generate straight line to plot """

    return [m * i + b for i in xt]

def predict (xtest, ytest, m, b):
    ypredict = abline (xtest, m, b)

    # Co-efficient of determination - R-squared
    SSR  = np.sum ( ( ypredict - np.mean(ytest) )**2 )
    SSE  = np.sum ( ( ytest    - ypredict       )**2 )
    SSTO = np.sum ( ( ytest    - np.mean(ytest) )**2 )

    r_squared = SSR / SSTO

    print ( 'SSR / SSTO' , r_squared, '1 - SSE / SSTO', 1 - SSE / SSTO )
     
#   for i in range(0,len(ytest)):
#       err_sum = print ytest[i], ypredict[i]
    return ypredict


filename = '../datasets/UCI/iris/iris.data'

df       = pd.read_csv(filename,sep=',')

# Attribute Information:
# 0. sepal length in cm
# 1. sepal width in cm
# 2. petal length in cm
# 3. petal width in cm
# 4. class: 
#    0 - Iris Setosa
#    1 - Iris Versicolour
#    2 - Iris Virginica

# Map label to integer

mapping  = {'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2}
df       =  df.replace({'Iris-setosa':mapping,'Iris-versicolor':mapping, 'Iris-virginica':mapping})
df.iloc[:, 4] = df.iloc[:, 4].apply(pd.to_numeric)

dataset = df.values

xt = dataset[:,0]  # Sepal Length
yt = dataset[:,3]  # petal Width

# Train

m_s, b_s   = simple_linear_regression (xt, yt)
m_l, b_l,c = least_squares_regression (xt, yt)

# Test

test_SepalLength  = np.array([5.1, 5.9, 6.9])
test_SepalWidth   = np.array([3.3, 3.0, 3.1])
test_PetalLength  = np.array([1.7, 4.2, 5.4])
test_PetalWidth   = np.array([0.5, 1.5, 2.1])

# Given Sepal Length predict Petal Width
predict_PetalWidth = predict (test_SepalLength, test_PetalWidth, m_s, b_s)


# plt.plot(xt,yt                  ,c='k', label='training data', marker='o',ls='')
colors = ['cyan','magenta','yellow']
import matplotlib
plt.scatter(xt, yt, c=dataset[:,4], cmap=matplotlib.colors.ListedColormap(colors))

plt.plot(xt,abline(xt, m_s, b_s),c='b', label='simple train')
plt.plot(xt,abline(xt, m_l, b_l),c='r', label='least square train')

plt.plot(test_SepalLength,test_PetalWidth,c='g',         label='test data',     marker='s', ls='--', )
plt.plot(test_SepalLength,predict_PetalWidth,c='orange', label='precit simple', marker='d', ls='')

plt.legend(loc=0)

plt.show()
