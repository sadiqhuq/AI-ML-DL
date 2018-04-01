# # Imports and Options
import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.float_format', lambda x: '%.2f' % x)


# # Read Data
train = pd.read_csv('./input/train.csv')

# train = train.drop("Id",1) # Remove Id column

# print('Features: \n', *list(train), sep='\t')

print ( 'Train - number of features:  ', train.shape[1] )
print ( 'Train - number of instances: ', train.shape[0] )

# # Data Check
# print ( train.head() )
# print ( train.describe() )

# # Use Correlation to identify 5 important features

train_corr = train.corr()
train_corr.sort_values(["SalePrice"], ascending = False, inplace = True)
print(train_corr.SalePrice[train_corr.SalePrice>0.5])
fig, ax = plt.subplots(figsize=(15,15))
sns.heatmap(train_corr, vmin=-1.0,vmax=1.0, square=True,cmap='RdBu',center=0);



# # Use PCA to identify 5 important features