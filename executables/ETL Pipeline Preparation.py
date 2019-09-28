#!/usr/bin/env python
# coding: utf-8

# # ETL Pipeline Preparation
# Follow the instructions below to help you create your ETL pipeline.
# ### 1. Import libraries and load datasets.
# - Import Python libraries
# - Load `messages.csv` into a dataframe and inspect the first few lines.
# - Load `categories.csv` into a dataframe and inspect the first few lines.

# In[1]:


# import necessary packages
import pandas as pd
from sqlalchemy import create_engine


# In[2]:


def read_data(msg_file='messages.csv', cat_file='categories.csv'):
    """
    Reads file paths of the two datasets to dataframes,merges on id
    Returns merged Pandas DataFrame
    :param msg_file: file path of message dataset
    :param cat_file: file path of categories dataset
    :return Pandas DataFrame
    """
    messages = pd.read_csv(msg_file)
    categories = pd.read_csv(cat_file)
    # merge datasets
    df = messages.merge(categories, on='id')
    return df, categories


# In[3]:


def process_data(df, categories):
    """
    Creates separate columns for 36 categories from categories DataFrame 
    Concat back to resultant DataFrame
    :param df: Messages and Categories DataFrame
    :param categories: Categories DataFrame
    :return df with categories encoded 
    """
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(';', expand=True)
    # select the first row of the categories dataframe
    row = categories.loc[0,:]

    # extracts a list of new column names for categories.
    category_colnames = row.str.slice(stop=-2).tolist()
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.slice(start=-1)
        # convert column from string to numeric
        categories[column] = categories[column].astype('int')
    
    # drop original column from merged DataFrame
    df.drop('categories', axis=1, inplace=True)
    # add splitted and renamed categories to final DataFrame
    df = pd.concat([df, categories], axis=1)
    # check duplicates
    print("{} duplicates found to be removed".format(df.duplicated().sum()))
    # drop duplicates
    df.drop_duplicates(inplace=True)
    return df


# In[4]:


def save_table(df, dbname, tablename):
    """
    Saves pandas DataFrame as specified tablename in the specified database 
    :param df: Messages and Categories DataFrame
    :param dbname: sql database name
    :param tablename: sql table name
    :return 
    """   
    engine = create_engine('sqlite:///{}'.format(dbname))
    engine.execute('DROP TABLE IF EXISTS {}'.format(tablename)) # drop if exists
    df.to_sql(tablename, engine, index=False)


# In[5]:


def read_table(dbname, tablename):
    """
    Reads specified tablename in the specified database to pandas DataFrame
    :param dbname: sql database name
    :param tablename: sql table name
    :return df: pandas DataFrame
    """       
    engine = create_engine('sqlite:///{}'.format(dbname))
    df =pd.read_sql_table(tablename, con=engine)
    return df


# In[6]:


df, categories = read_data()
df = process_data(df, categories)
save_table(df, 'disaster.db', 'messages')


# In[7]:


# read_table('disaster.db', 'messages')

