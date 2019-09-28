#!/usr/bin/env python
# coding: utf-8

# # ETL Pipeline Preparation
#  import necessary packages
import pandas as pd
import sys
from sqlalchemy import create_engine


def read_data(msg_file, cat_file):
    """
    Reads file paths of the two datasets to dataframes,merges on id
    Returns merged Pandas DataFrame
    :param msg_file: file path of message dataset
    :param cat_file: file path of categories dataset
    :return Pandas DataFrame
    """
    messages = pd.read_csv(msg_file)
    categories = pd.read_csv(cat_file)
    #  merge datasets
    df = messages.merge(categories, on='id')
    print('Messages and Categories DataFrames are read and merged.')
    return df


def process_data(df):
    """
    Creates separate columns for 36 categories
    Concat back to resultant DataFrame
    :param df: Messages and Categories DataFrame
    :return df with categories encoded
    """
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(';', expand=True)
    # select the first row of the categories dataframe
    row = categories.loc[0, :]

    # extracts a list of new column names for categories.
    category_colnames = row.str.slice(stop=-2).tolist()
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.slice(start=-1)
        # convert column from string to numeric
        categories[column] = categories[column].astype('int')

    #  drop original column from merged DataFrame
    df.drop('categories', axis=1, inplace=True)
    #  add splitted and renamed categories to final DataFrame
    df = pd.concat([df, categories], axis=1)
    # check duplicates
    print("Record Count : {}".format(df.shape[0]))
    print("{} duplicates found to be removed".format(df.duplicated().sum()))
    # drop duplicates
    df.drop_duplicates(inplace=True)
    print("Final Record Count : {}".format(df.shape[0]))
    return df


def save_table(df, dbname, tablename='messages'):
    """
    Saves pandas DataFrame as specified tablename in the specified database
    :param df: Messages and Categories DataFrame
    :param dbname: sql database name
    :param tablename: sql table name
    :return
    """
    engine = create_engine('sqlite:///{}'.format(dbname))
    engine.execute('DROP TABLE IF EXISTS {}'.format(tablename))
    df.to_sql(tablename, engine, index=False)
    print("{} Table is saved to : {}".format(tablename, dbname))


def read_table(dbname, tablename):
    """
    Reads specified tablename in the specified database to pandas DataFrame
    :param dbname: sql database name
    :param tablename: sql table name
    :return df: pandas DataFrame
    """
    engine = create_engine('sqlite:///{}'.format(dbname))
    df = pd.read_sql_table(tablename, con=engine)
    return df

def main():
    
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

    
        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = read_data(msg_file=messages_filepath, cat_file=categories_filepath)

        print('Cleaning data...')
        df = process_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_table(df, dbname=database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    """
    Main functionality
    """
    # Example Execution:
    # python process_data.py messages.csv categories.csv DisasterResponse.db
    main()

