from pathlib import Path
from typing import List
import pandas as pd
import numpy as np
from assignments.assignment1.a_load_file import read_dataset


##############################################
# Example(s). Read the comments in the following method(s)
##############################################
def pandas_profile(df: pd.DataFrame, result_html: str = 'report.html'):
    """
    This method will be responsible to extract a pandas profiling report from the dataset.
    Do not change this method, but run it and look through the html report it generated.
    Always be sure to investigate the profile of your dataset (max, min, missing values, number of 0, etc).
    """
    from pandas_profiling import ProfileReport

    profile = ProfileReport(df, title="Pandas Profiling Report")
    if result_html is not None:
        profile.to_file(result_html)
    return profile.to_json()


##############################################
# Implement all the below methods
# All methods should be dataset-independent, using only the methods done in the assignment
# so far and pandas/numpy/sklearn for the operations
##############################################
def get_column_max(df: pd.DataFrame, column_name: str) -> float:
    #dropping NA from the column so that we dont get max as NA
    column_array = np.array(df[column_name].dropna())
    if (len(column_array) == 0):
        return np.nan
    else:
        return np.max(column_array)


def get_column_min(df: pd.DataFrame, column_name: str) -> float:
    # dropping NA from the column so that we dont get min as NA
    column_array = np.array(df[column_name].dropna())
    if (len(column_array) == 0):
        return np.nan
    else:
        return np.min(column_array)


def get_column_mean(df: pd.DataFrame, column_name: str) -> float:
    column_array = np.array(df[column_name].dropna())
    # checking if the column is empty after removing nan
    if (len(column_array) == 0):
        return np.nan
    else:
        return np.mean(column_array)


def get_column_count_of_nan(df: pd.DataFrame, column_name: str) -> float:
    return df[column_name].isna().sum()


def get_column_number_of_duplicates(df: pd.DataFrame, column_name: str) -> float:
    return len(df[column_name].dropna())-len(df[column_name].dropna().unique())


def get_numeric_columns(df: pd.DataFrame) -> List[str]:
    # getting all numeric list include int, float, binary.. etc
    allnumericlist = df.select_dtypes(include=np.number, exclude=np.bool).columns.tolist()
    boolist = get_binary_columns(df)
    numericlist = []
    # removing binary from the numeric
    for name in allnumericlist:
        if name not in boolist:
            numericlist.append(name)
    return numericlist


def get_binary_columns(df: pd.DataFrame) -> List[str]:
    boollist = []
    for col in df:
        #checking if the columns only contain 0,1 elements
        if (np.isin(df[col].dropna().unique(), [1, 0]).all()):
            boollist.append(col)
    return boollist


def get_text_categorical_columns(df: pd.DataFrame) -> List[str]:
    #categorial columns are something that contains dtype as object
    allcategoricallist = df.select_dtypes(include=np.object, exclude=np.bool).columns.tolist()
    boolist = get_binary_columns(df)
    categoriallist = []
    # removing binary from the categorical
    for name in allcategoricallist:
        if name not in boolist:
            categoriallist.append(name)
    return categoriallist


def get_correlation_between_columns(df: pd.DataFrame, col1: str, col2: str) -> float:
    return (df[col1].corr(df[col2],method="pearson"))


if __name__ == "__main__":
    df = read_dataset(Path('..', '..', 'iris.csv'))
    a = pandas_profile(df)
    assert get_column_max(df, df.columns[0]) is not None
    assert get_column_min(df, df.columns[0]) is not None
    assert get_column_mean(df, df.columns[0]) is not None
    assert get_column_count_of_nan(df, df.columns[0]) is not None
    assert get_column_number_of_duplicates(df, df.columns[0]) is not None
    assert get_numeric_columns(df) is not None
    assert get_binary_columns(df) is not None
    assert get_text_categorical_columns(df) is not None
    assert get_correlation_between_columns(df, df.columns[0], df.columns[1]) is not None
