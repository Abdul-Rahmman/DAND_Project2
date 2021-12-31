# -*- coding: utf-8 -*-
"""

@author:Manee44
"""

# import libraries

import pandas as pd
import numpy as np
import sqlite3
from sqlalchemy import create_engine


# load messages dataset
messages = pd.read_csv('./Data/messages.csv')

# load categories dataset
categories = pd.read_csv('./Data/categories.csv')

# merge datasets
df = pd.merge(messages, categories,  how='outer', on='id')

# create a dataframe of the 36 individual category columns
categories = df.categories.str.split(';',expand=True)


# select the first row of the categories dataframe
row = categories.head(1)

# use this row to extract a list of new column names for categories.
# one way is to apply a lambda function that takes everything 
# up to the second to last character of each string with slicing
category_colnames = row.applymap(lambda x: x[:-2]).iloc[0,:]

# rename the columns of `categories`
categories.columns = category_colnames

for column in categories:
    # set each value to be the last character of the string
    categories[column] = categories[column].astype(str).str[-1]
    
    # convert column from string to numeric
    categories[column] = categories[column].astype(int)
    
# drop the original categories column from `df`
del df['categories']

# concatenate the original dataframe with the new `categories` dataframe
df = pd.concat([df,categories], axis = 1, join = 'inner' )

# drop duplicates
df.drop_duplicates(inplace=True)

engine = create_engine('sqlite:///DataBase.db')
df.to_sql('DataBase', engine, index=False)





    
    