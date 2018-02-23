#!/usr/bin/python3

""" 
Sadiq Huq, Karlsruhe Institute of Technology

References:
Andrew Ng, Stanford - Coursera
Numerical Recipies (2007)

Fitting Data to a Straight Line 
y(x) = y(x|m,b) = mx + b

Machine Learning Notation
h_theta(x_i) = theta_0 + theta_1 * x_i

h: hypothesis that maps x => y
m: training examples
x: independent variable
y: dependent variable / label / target value
co-efficient: theta_0 (intercept), theta_1 (slope)
Minimize  half the function: h_theta (x_i) - y_i
avergaed MSE/2: cost function J(theta_0, theta_1):
0.5 * m * sum(h_theta (x_i) - y_i )^2

Goal: minmize J(thet_0, Theta_1)

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def simple_linear_regression(x, y):
    """
    General Linear Least Squares
    Find the slope and intercept using
    mean(x), mean(y), covar(x,y), var(x)
    https://en.wikipedia.org/wiki/Simple_linear_regression#Numerical_example
    http://mathworld.wolfram.com/LeastSquaresFitting.html
    """
    # norm = len(xt)-1   # NumPy normalizes variance by N-ddof 
    norm = 0
    
    covar = np.cov(x,y,ddof=norm)
 
    # theta_1 = covar [0,1] / np.var(x,ddof=norm)
    theta_1 = covar [0,1] / covar[0,0]
    theta_0 = np.mean(y) - ( theta_1 * np.mean(x) )

    # The propotion of SSyy (covar[1,1]) acounted for by the regresion
    print ( 'r-squared:', covar[0,1]**2 / ( covar[0,0]*covar[1,1] ) )

    return theta_0, theta_1, 0

def gradient_descent_regression(x, y):
    niters    = 100
    learnrate = 0.001

    N = y.size

    theta_0 = 0
    theta_1 = 0
    cost    = 0

    return theta_0, theta_1, cost

def abline (xt, theta_0, theta_1):
    """ Generate discrete straight line to plot """

    return [theta_0 + theta_1*i for i in xt]

def predict (xtest, ytest, theta_0, theta_1):
    ypredict = abline (xtest, theta_0, theta_1)

    # Co-efficient of determination - R-squared
    SSR  = np.sum ( ( ypredict - np.mean(ytest) )**2 )
    SSE  = np.sum ( ( ytest    - ypredict       )**2 )
    SSTO = np.sum ( ( ytest    - np.mean(ytest) )**2 )

    r_squared = SSR / SSTO

    print ( 'SSR / SSTO: ' , r_squared, '1 - SSE / SSTO: ', 1 - SSE / SSTO )
     
    RMSE = np.sqrt ( SSE / len(ytest) )
    print ( 'RMSE: ', RMSE )

    return ypredict

def run():
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

    # Label string to int
    # df.iloc[:, 4] = df.iloc[:, 4].apply(pd.to_numeric)

    # # Scatter plot of the dataframe
    # multi = pd.plotting.scatter_matrix(df, c=df.iloc[:, 4], figsize=(15, 15), marker='o',
    #                            hist_kwds={'bins': 20}, s=60, alpha=.8)


    dataset = df.values

    xt = dataset[:,0]  # Sepal Length
    yt = dataset[:,3]  # petal Width

    # Train

    theta_0_s, theta_1_s, cost = simple_linear_regression    (xt, yt)
    theta_0_g, theta_1_g, cost = gradient_descent_regression (xt, yt)

    # Test

    test_SepalLength  = np.array([5.1, 5.9, 6.9])
    test_SepalWidth   = np.array([3.3, 3.0, 3.1])
    test_PetalLength  = np.array([1.7, 4.2, 5.4])
    test_PetalWidth   = np.array([0.5, 1.5, 2.1])

    # Given Sepal Length predict Petal Width
    predict_PetalWidth = predict (test_SepalLength, test_PetalWidth, theta_0_s, theta_1_s)


    # plt.plot(xt,yt                  ,c='k', label='training data', marker='o',ls='')
    colors = ['cyan','magenta','yellow']
    import matplotlib
    plt.scatter(xt, yt, c=dataset[:,4], cmap=matplotlib.colors.ListedColormap(colors))

    plt.plot(xt,abline(xt, theta_0_s, theta_1_s),c='b', label='simple train')
    plt.plot(xt,abline(xt, theta_0_g, theta_1_g),c='r', label='least square train')

    plt.plot(test_SepalLength,test_PetalWidth,c='g',         label='test data',     marker='s', ls='--', )
    plt.plot(test_SepalLength,predict_PetalWidth,c='orange', label='predict simple', marker='d', ls='')

    plt.legend(loc=0)

    plt.show()

if __name__ == '__main__':
   run()
