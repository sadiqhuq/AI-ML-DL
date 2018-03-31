import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('./input/train.csv')

nrows, ncols = df.shape

print ( 'Number of features:  ', ncols-1 )  # Excluding Id
print ( 'Number of instances: ', nrows   )  

features = list(df)
print('Features: \n', *features, sep='\t')

# print ( df.head() )
# print ( df.describe() )

