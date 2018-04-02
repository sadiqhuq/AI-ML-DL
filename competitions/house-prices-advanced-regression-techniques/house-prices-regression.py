# # Imports and Options
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error, make_scorer
 
pd.set_option('display.float_format', lambda x: '%.2f' % x)


# # Read Data
train_df = pd.read_csv('./input/train.csv')
test_df  = pd.read_csv('./input/test.csv')

ntrain   = train_df.shape[0]
print ( 'Number of features:  ', train_df.shape[1] )
print ( 'Number of instances: ', train_df.shape[0] )

train_SalePrice = train_df.SalePrice

train_df.drop("SalePrice", axis = 1, inplace = True) # Remove Id column

# train_df.drop("Id", axis = 1, inplace = True) # Remove Id column
# test_df.drop("Id", axis = 1, inplace = True)

# print('Features: \n', *list(train), sep='\t')

print ( '\nNumber of features:  ', train_df.shape[1] )
print ( 'Number of instances: ', train_df.shape[0] )

print ( '\nNumber of features:  ', test_df.shape[1] )
print ( 'Number of instances: ', test_df.shape[0] )

merged_df = pd.concat([train_df, train_df], axis = 0)

del (train_df, test_df)

# # Data Check

# # Gap Filling
# # Refer: https://www.kaggle.com/juliencs/a-study-on-regression-applied-to-the-ames-dataset
feat_cat  = merged_df.select_dtypes(include = ["object"]).columns
feat_num  = merged_df.select_dtypes(exclude = ["object"]).columns

merged_cat = merged_df[feat_cat]
merged_num = merged_df[feat_num]

merged_cat = pd.get_dummies(merged_cat)
merged_num = merged_num.fillna(merged_num.median())

merged     = pd.concat([merged_num, merged_cat], axis = 1)

print ( '\nCheck datatypes of features: ' )
print ( merged.columns.to_series().groupby(merged.dtypes).groups )

train      = pd.DataFrame(merged[0:ntrain])
test       = pd.DataFrame(merged[ntrain+1:])

# #  Seletect Features

predictors =  list(merged) #  Use all features as precictors
# predictors = ['LotArea']
# predictors = predictors[6:9]
# predictors.remove('SalePrice')

# # Build Model with all features

# train_X.SalePrice = np.log1p(train.SalePrice)

train_X, vald_X, train_y, vald_y = train_test_split(train[predictors], 
                                                  np.log1p(train_SalePrice),
                                                  test_size = 0.3,
                                                  random_state = 0)

# train_X.head()
# val_X.head()

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

print( '\nvalidation MAE LRG: %.4f' % mean_absolute_error(vald_y, predicted_LRG) )
print( 'validation MAE RFG: %.4f' % mean_absolute_error(vald_y, predicted_RFG) )
print( 'validation MAE DTR: %.4f' % mean_absolute_error(vald_y, predicted_DTR)  )


# # RMSE

def rmse_cv(model,X,y):
    scorer = make_scorer(mean_squared_error, greater_is_better = False)
    rmse= np.sqrt(-cross_val_score(model, X, y, scoring = scorer, cv = 4,n_jobs=2))
    return(rmse.mean())
    
print("\nRMSE train data")
print("Linear Regression:", rmse_cv(model_LRG, train_X, train_y))
print("Random Forest    :", rmse_cv(model_RFG, train_X, train_y))
print("Decission Tree   :", rmse_cv(model_DTR, train_X, train_y))

print("\nRMSE validation data")
print("Linear Regression:", rmse_cv(model_LRG, vald_X, predicted_LRG))
print("Random Forest    :", rmse_cv(model_RFG, vald_X, predicted_RFG))
print("Decission Tree   :", rmse_cv(model_DTR, vald_X, predicted_DTR))
