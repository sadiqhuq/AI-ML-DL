# # Imports and Options
import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

pd.set_option('display.float_format', lambda x: '%.2f' % x)


# # Read Data
train = pd.read_csv('./input/train.csv')

# train = train.drop("Id",1) # Remove Id column

# print('Features: \n', *list(train), sep='\t')

print ( 'Number of features:  ', train.shape[1] )
print ( 'Number of instances: ', train.shape[0] )

# # Data Check
# print ( train.head() )
# print ( train.describe() )

# # Use PCA to identify 5 important features


