# -*- coding: utf-8 -*-
"""

@author:Manee44
"""

# import libraries
import sys
import pandas as pd
import numpy as np
import sqlite3
from sqlalchemy import create_engine

def transform(messages_filepath, categories_filepath):
    # load messages dataset
    messages = pd.read_csv(messages_filepath)

    # load categories dataset
    categories = pd.read_csv(categories_filepath)

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
    categories.replace(2, 1, inplace=True)
       # drop the original categories column from `df`
    del df['categories']

       # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories], axis = 1, join = 'inner' )

    return df

def clean(df):   
    # drop duplicates
    df.drop_duplicates(inplace=True)
    return df

def save_data(df,db_name):
    engine = create_engine('sqlite:///'+db_name)
    df.to_sql('Disater_Response', engine, if_exists = 'replace', index=False)
    
    
    
    
def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please make sure the provided filepath is in correct order.')


if __name__ == '__main__':
    main()
