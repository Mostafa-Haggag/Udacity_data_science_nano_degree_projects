import sys
import pandas as pd
import numpy as np
import os
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
"""
Description of the function:
Load the data from csv to data frame
Parameters:
messages_filepath The path of the message csv
categories_filepath the path of the cateogire ccsv
Returns:
df return a merger data frame of both merging ID
"""
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories,how='inner',on=['id'])
    return df

def clean_data(df):
"""
Description of the function:
Takes the data frame and clean it by checking for nulls and dupplicates
Parameters:
df takes as an input the main data frame

Returns:
df return the cleaned data frame

"""
    categories = df["categories"].str.split(';',expand=True)
    row = categories.iloc[[1]]
    category_colnames = [category_name.split('-')[0] for category_name in row.values[0]]
    for column in categories:
        # set each value to be the last character of the string
        categories[column] =  categories[column].astype(str).str[-1:]
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    categories.columns = category_colnames
    df = pd.concat([df,categories], join='inner', axis=1)
    df.drop(['categories'], axis=1, inplace=True)
    df = df.drop(['child_alone'],axis = 1)
    df.drop_duplicates(inplace=True)
    df['related']=df['related'].map(lambda x: 0 if x == 2 else x)
    print('Duplicates remaining:', df.duplicated().sum())
    return df
def save_data(df, database_filename):
"""
Description of the function:
saving the data after it wa cleaned
Parameters:
df data frame
database_file name the new name of the data frame that we will save as
Returns:
no return

"""
    engine = create_engine('sqlite:///' + database_filename)
    table_name = os.path.basename(database_filename).replace(".db","") + "_table"
    df.to_sql(table_name, engine, index=False, if_exists='replace')
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
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()