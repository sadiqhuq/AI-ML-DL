# # Imports and Options
import numpy  as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error, make_scorer




 
pd.set_option('display.float_format', lambda x: '%.2f' % x)


# # Read Data
train = pd.read_csv('./input/train.csv')
train.drop("Id", axis = 1, inplace = True) # Remove Id column

# print('Features: \n', *list(train), sep='\t')

print ( 'Number of features:  ', train.shape[1] )
print ( 'Number of instances: ', train.shape[0] )

# # Data Check
# print ( train.head() )
# print ( train.describe() )

# # Gap Filling
# # Refer: https://www.kaggle.com/juliencs/a-study-on-regression-applied-to-the-ames-dataset
feat_cat  = train.select_dtypes(include = ["object"]).columns
feat_num  = train.select_dtypes(exclude = ["object"]).columns

train_cat = train[feat_cat]
train_num = train[feat_num]

train_num = train_num.fillna(train_num.median())
train_cat = pd.get_dummies(train_cat)

train     = pd.concat([train_num, train_cat], axis = 1)

# #  Seletect Features

# # Use CORR to identify important features
# 
# train_corr = train.corr()
# train_corr.sort_values(["SalePrice"], ascending = False, inplace = True)
# predictors = list(train_corr.SalePrice[train_corr.SalePrice>0.5].index)
# predictors.remove('SalePrice');
# print('Selected Predictors: ', predictors)

# predictors =  ['LotArea']

predictors =  list(train) #  Use all features as precictors
# predictors.remove('SalePrice')

# # Build Model with the 5 important features

train.SalePrice = np.log1p(train.SalePrice)

train_X, vald_X, train_y, vald_y = train_test_split(train[predictors], 
                                                  train.SalePrice,
                                                  test_size = 0.3,
                                                  random_state = 0)

# # Linear Regression

model_LRG = LinearRegression()
model_LRG.fit(train_X, train_y)

# # Random Forest

model_RFG = RandomForestRegressor(random_state=0)
model_RFG.fit(train_X , train_y)

# # Decission Tree

max_leaf_nodes = 530     # Found by y trial and error
model_DTR = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes,random_state=0)
model_DTR.fit(train_X , train_y)

# # Evaluate Model with Validation Data

predicted_LRG  = model_RFG.predict(vald_X[predictors])
predicted_RFG  = model_RFG.predict(vald_X[predictors])
predicted_DTR  = model_DTR.predict(vald_X[predictors])

print( 'validation MAE LRG: %.4f' % mean_absolute_error(vald_y, predicted_LRG) )
print( 'validation MAE RFG: %.4f' % mean_absolute_error(vald_y, predicted_RFG) )
print( 'validation MAE DTR: %.4f' % mean_absolute_error(vald_y, predicted_DTR)  )


# # RMSE

def rmse_cv(model,X,y):
    scorer = make_scorer(mean_squared_error, greater_is_better = False)
    rmse= np.sqrt(-cross_val_score(model, X, y, scoring = scorer, cv = 4,n_jobs=2))
    return(rmse.mean())
    
print("RMSE train data")
print("Linear Regression:", rmse_cv(model_LRG, train_X, train_y))
print("Random Forest    :", rmse_cv(model_RFG, train_X, train_y))
print("Decission Tree   :", rmse_cv(model_DTR, train_X, train_y))

print("RMSE validation data")
print("Linear Regression:", rmse_cv(model_LRG, vald_X, predicted_LRG))
print("Random Forest    :", rmse_cv(model_RFG, vald_X, predicted_RFG))
print("Decission Tree   :", rmse_cv(model_DTR, vald_X, predicted_DTR))