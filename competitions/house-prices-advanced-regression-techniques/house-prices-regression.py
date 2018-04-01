# # Imports and Options
import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import svm
from sklearn.metrics import mean_absolute_error, make_scorer

 
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
feat_cat = train.select_dtypes(include = ["object"]).columns
feat_num = train.select_dtypes(exclude = ["object"]).columns

train_cat = train[feat_cat]
train_num = train[feat_num]

train_num = train_num.fillna(train_num.median())
train_cat = pd.get_dummies(train_cat)

train = pd.concat([train_num, train_cat], axis = 1)

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

train_X, val_X, train_y, val_y = train_test_split(train[predictors], 
                                                  train.SalePrice,
                                                  test_size = 0.3,
                                                  random_state = 0)

# # Random Forest

RFG_model = RandomForestRegressor(random_state=0)
RFG_model.fit(train_X , train_y)

# # Decission Tree

max_leaf_nodes = 530     # Found by y trial and error
DT_model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes,random_state=0)
DT_model.fit(train_X , train_y)

# # Evaluate Model with Validation Data

test = pd.read_csv('./input/test.csv')

RFG_predicted = RFG_model.predict(val_X[predictors])
DT_predicted = DT_model.predict(val_X[predictors])


print( 'validation MAE RFG: %.4f' % mean_absolute_error(val_y, RFG_predicted) )
print( 'validation MAE DT:  %.4f' % mean_absolute_error(val_y, DT_predicted)  )

# # Apply Model to Test Data


# Prepare to submit

# submit = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_dep})
# submit.to_csv('submission.csv', index=False)

