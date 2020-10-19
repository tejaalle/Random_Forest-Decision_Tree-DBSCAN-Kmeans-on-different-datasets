import collections
from pathlib import Path
from typing import Union, Optional
from enum import Enum

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

from assignments.assignment1.b_data_profile import *
from assignments.assignment1.c_data_cleaning import *
from assignments.assignment1.a_load_file import read_dataset
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


##############################################
# Example(s). Read the comments in the following method(s)
##############################################


##############################################
# Implement all the below methods
# All methods should be dataset-independent, using only the methods done in the assignment
# so far and pandas/numpy/sklearn for the operations
##############################################
def generate_label_encoder(df_column: pd.Series) -> LabelEncoder:
    labelEncoder = LabelEncoder()
    #passing the column to labelEncoder
    labelEncoder.fit_transform(df_column)
    return labelEncoder


def generate_one_hot_encoder(df_column: pd.Series) -> OneHotEncoder:
    hotEncoder = OneHotEncoder()
    #hotencoder accepts 2-D array so converting the column into 2D array
    hotEncoder.fit(pd.DataFrame(df_column))
    return hotEncoder


def replace_with_label_encoder(df: pd.DataFrame, column: str, le: LabelEncoder) -> pd.DataFrame:
    df[column] = le.fit_transform(df[column])
    return df


def replace_with_one_hot_encoder(df: pd.DataFrame, column: str, ohe: OneHotEncoder, ohe_column_names: List[str]) -> pd.DataFrame:
    columnNames = []
    # taking column names into list expect the one which is being encoded
    for i in df.columns:
        if i!= column:
            columnNames.append(i)
    #using columnTransformer with one hot encoder passed as a argument to apply for one column and remaining column will just passthrough
    # columnTransformer = ColumnTransformer([('encoder',ohe,[column])],remainder='passthrough')
    # df1 = np.array(columnTransformer.fit_transform(df),dtype = np.object)
    # le = generate_label_encoder(df[column])
    # df = replace_with_label_encoder(df,column,le)
    enc_df = pd.DataFrame(ohe.fit_transform(df[[column]]).toarray())
    df=df.drop(column,axis=1)
    df = enc_df.join(df)
    #combining the newly encoded column names with the old ones expect encoded one
    columnNames=ohe_column_names+columnNames
    df.columns = columnNames
    return df


def replace_label_encoder_with_original_column(df: pd.DataFrame, column: str, le: LabelEncoder) -> pd.DataFrame:
    df[column]= le.inverse_transform(df[column])
    return df


def replace_one_hot_encoder_with_original_column(df: pd.DataFrame,
                                                 columns: List[str],
                                                 ohe: OneHotEncoder,
                                                 original_column_name: str) -> pd.DataFrame:
    df1 = df
    #taking only the columns which are to be replaced and dropping remaining columns
    for i in df.columns:
        if i not in columns:
            df = df.drop(i,axis=1)
    orginal= ohe.inverse_transform(df)
    original_items = []
    #taking the original column items into a list
    for item in orginal:
        original_items.append(item[0])
    for i in df1.columns:
        if i in columns:
            df1 = df1.drop(i,axis=1)
    #appending the items to the dataframe column
    df1[original_column_name]=original_items

    return df1


if __name__ == "__main__":
    df = pd.DataFrame({'a':[1,2,3,4], 'b': [True, True, False, False], 'c': ['one', 'two', 'three', 'four']})
    # le = generate_label_encoder(df.loc[:, 'c'])
    # assert le is not None
    ohe = generate_one_hot_encoder(df.loc[:, 'c'])
    # assert ohe is not None
    # assert replace_with_label_encoder(df, 'c', le) is not None
    assert replace_with_one_hot_encoder(df, 'c', ohe, list(ohe.get_feature_names())) is not None
    # assert replace_label_encoder_with_original_column(replace_with_label_encoder(df, 'c', le), 'c', le) is not None
    assert replace_one_hot_encoder_with_original_column(replace_with_one_hot_encoder(df, 'c', ohe, list(ohe.get_feature_names())),
                                                        list(ohe.get_feature_names()),
                                                        ohe,
                                                        'c') is not None
    print("ok")
