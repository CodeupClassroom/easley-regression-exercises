import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import env
import acquire

datapath = env.datapath

def acquire_grades():
    '''
    Grab our data from a path, read in from csv
    '''
    df = pd.read_csv(datapath+'student_grades.csv')
    return df

def clean_grades(df):
    '''
    Takes in a df of student exam grades and cleans the data appropriately by dropping null values,
    replacing whitespace,
    and converting data to numerical data types
    as well as dropping student_id column from the dataframe
    
    return: df, a cleaned pandas dataframe
    '''
    df['exam3'] = df['exam3'].replace(r'^\s*$', np.nan, regex=True)
    df = df.dropna()
    df['exam3'] = df['exam3'].astype('int')
    df['exam1'] = df['exam1'].astype('int')
    df = df.drop(columns='student_id')
    return df

def split_data(df):
    '''
    split our data,
    takes in a pandas dataframe
    returns: three pandas dataframes, train, test, and validate
    '''
    train_val, test = train_test_split(df, train_size=0.8, random_state=1349)
    train, validate = train_test_split(train_val, train_size=0.7, random_state=1349)
    return train, validate, test

def wrangle_grades():
    '''
    wrangle_grades will read in our student grades as a pandas dataframe,
    clean the data
    split the data
    return: train, validate, test sets of pandas dataframes from student grades, stratified on final_grade
    '''
    df = clean_grades(acquire_grades())
    return split_data(df)

def clean_telco(df):
    '''
    clean the two year contract data down to 
    monthly charges, total charges, tenure, and customer id,
    replace whitespace values in total charges with zeros where appropriate
    '''
    df = df[['customer_id', 'monthly_charges', 'tenure', 'total_charges']]
    df['total_charges'] = df['total_charges'].replace(r'^\s*$', np.nan, regex=True)
    df = df.fillna(0)
    df['total_charges'] = df['total_charges'].astype('float')
    return df
    
def wrangle_telco():
    '''
    wrangle_telco will read in our telco data for two year contract customers via a sql query, clean the data down to 
    monthly charges, total charges, tenure, and customer id,
    replace whitespace values in total charges with zeros where appropriate
    and then split the data
    
    return: train, validate, and test sets of telco data
    '''
    df = clean_telco(acquire.get_telco_data())
    return split_data(df)
    
    