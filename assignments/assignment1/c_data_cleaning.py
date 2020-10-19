import collections
from pathlib import Path
from typing import Union, Optional
from enum import Enum

import pandas as pd
import numpy as np

from assignments.assignment1.b_data_profile import *
from assignments.assignment1.a_load_file import read_dataset


##############################################
# Example(s). Read the comments in the following method(s)
##############################################
class WrongValueNumericRule(Enum):
    """
    You'll use these enumeration possibilities in your implemented methods below
    """
    MUST_BE_POSITIVE = 0
    MUST_BE_NEGATIVE = 1
    MUST_BE_GREATER_THAN = 2
    MUST_BE_LESS_THAN = 3


class DistanceMetric(Enum):
    """
    You'll use these enumeration possibilities in your implemented methods below
    """
    EUCLIDEAN = 0
    MANHATTAN = 1


##############################################
# Implement all the below methods
# All methods should be dataset-independent, using only the methods done in the assignment
# so far and pandas/numpy/sklearn for the operations
##############################################
def fix_numeric_wrong_values(df: pd.DataFrame,
                             column: str,
                             must_be_rule: WrongValueNumericRule,
                             must_be_rule_optional_parameter: Optional[float] = None) -> pd.DataFrame:
    #checking the rule using if condition
    if(must_be_rule==WrongValueNumericRule.MUST_BE_POSITIVE):
        #replacing negative elements and zero to nan
        df[column]=np.where(df[column]>0,df[column],np.nan)
    elif(must_be_rule==WrongValueNumericRule.MUST_BE_NEGATIVE):
        # replacing positive elements and zero to nan
        df[column] = np.where(df[column]<0, df[column], np.nan)
    elif(must_be_rule==WrongValueNumericRule.MUST_BE_GREATER_THAN):
        # replacing elements less than or equal to given parameter
        df[column] = np.where(df[column] > must_be_rule_optional_parameter, df[column], np.nan)
    elif(must_be_rule==WrongValueNumericRule.MUST_BE_LESS_THAN):
        # replacing elements greater than or equal to given parameter
        df[column] = np.where(df[column]<must_be_rule_optional_parameter, df[column], np.nan)
    return df


def fix_outliers(df: pd.DataFrame, column: str) -> pd.DataFrame:
    #getting numeric columns to apply outliners
    numeric = get_numeric_columns(df)
    categorial = get_text_categorical_columns(df)
    if (column in numeric):
        # Implement quartile based flooring and capping
        # This process is taken from "https://kanoki.org/2020/04/23/how-to-remove-outliers-in-python/" and "https://www.pluralsight.com/guides/cleaning-up-data-from-outliers"
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3-Q1
        min = Q1 - 1.5*IQR
        max = Q3 + 1.5*IQR
        #instead of deleting, replacing elements which are greater than Q3 with Q3 and less than Q1 with Q1 so that all are with in the outliners
        df[column] = np.where(df[column] < min, min, df[column])
        df[column] = np.where(df[column] > max, max, df[column])
        return df
    elif(column in categorial):
        x=df[column].unique()
        df2 =df.copy()
        #removing only one occurances of a Value
        for i in x:
            if(len(df[df[column]==i])==1):
                df[column].replace({i:np.nan},inplace = True)
        #if all the values are one occurances then returning original or else returning new one
        if(len(df.dropna())==0):
            return df2
        else:
            return df
    else:
        #returning same for binary and date
        return df


def fix_nans(df: pd.DataFrame, column: str) -> pd.DataFrame:
    #finding if column contains all nans
    if(get_column_count_of_nan(df,column)==len(df[column])):
        #removing the column with allnan as thier is no information in it and returning the dataframe
        df = df.drop(column,axis=1)
        return df
    numeric = get_numeric_columns(df)
    if column in numeric:
        # got idea from "https://towardsdatascience.com/data-handling-using-pandas-cleaning-and-processing-3aa657dc9418"
        #filling the Na with mean so that thoer wont be any differenvce with the added value
        df[column].fillna(get_column_mean(df,column), inplace=True)

    else:
        df[column].fillna(method="ffill", inplace=True)
    return df


def normalize_column(df_column: pd.Series) -> pd.Series:
    if(df_column.dtype=="object"):
        return df_column
    #Formula taken from "https://scikit-learn.org/stable/modules/preprocessing.html#scaling-features-to-a-range"
    new_df = (df_column - df_column.min()) / (df_column.max() - df_column.min())
    return pd.Series(new_df)


def standardize_column(df_column: pd.Series) -> pd.Series:
    #Formula taken from "https://medium.com/@rrfd/standardize-or-normalize-examples-in-python-e3f174b65dfc"
    # if(len(df_column.unique())==1):
    #     new_df = df_column-df_column
    #     return pd.Series(new_df)
    new_df = (df_column-df_column.mean())/df_column.std()
    return pd.Series(new_df)


def calculate_numeric_distance(df_column_1: pd.Series, df_column_2: pd.Series, distance_metric: DistanceMetric) -> pd.Series:
    df1 = np.array(df_column_1)
    df2 =np.array(df_column_2)
    if distance_metric==DistanceMetric.EUCLIDEAN:
        # applying euclidean distance formula
        df3 = np.zeros([len(df1)])
        for i in range(len(df1)):
            df3[i] = np.sqrt((df1[i]-df2[i])**2)
        return pd.Series(df3)
    else:
        # applying manhattan distance formula
        df3 = np.zeros([len(df1)])
        for i in range(len(df1)):
            df3[i] = np.abs((df1[i] - df2[i]))
        return pd.Series(df3)


def calculate_binary_distance(df_column_1: pd.Series, df_column_2: pd.Series) -> pd.Series:
    #applied hamming distance formula
    df1 = np.array(df_column_1)
    df2 = np.array(df_column_2)
    df3 = np.zeros([len(df1)])
    for i in range(len(df2)):
        if(df1[i]==df2[i]):
            df3[i]=0
        else:
            df3[i]=1
    return pd.Series(df3)


if __name__ == "__main__":
    df = pd.DataFrame({'a':[1,2,3,None], 'b': [True, True, False, None], 'c': ['one', 'two', np.nan, None]})
    assert fix_numeric_wrong_values(df, 'a', WrongValueNumericRule.MUST_BE_LESS_THAN, 2) is not None
    assert fix_numeric_wrong_values(df, 'a', WrongValueNumericRule.MUST_BE_GREATER_THAN, 2) is not None
    assert fix_numeric_wrong_values(df, 'a', WrongValueNumericRule.MUST_BE_POSITIVE, 2) is not None
    assert fix_numeric_wrong_values(df, 'a', WrongValueNumericRule.MUST_BE_NEGATIVE, 2) is not None
    assert fix_outliers(df, 'b') is not None
    assert fix_nans(df, 'c') is not None
    assert normalize_column(df.loc[:, 'a']) is not None
    assert standardize_column(df.loc[:, 'a']) is not None
    assert calculate_numeric_distance(df.loc[:, 'a'], df.loc[:, 'a'], DistanceMetric.EUCLIDEAN) is not None
    assert calculate_numeric_distance(df.loc[:, 'a'], df.loc[:, 'a'], DistanceMetric.MANHATTAN) is not None
    assert calculate_binary_distance(df.loc[:, 'b'], df.loc[:, 'b']) is not None
    print("ok")
