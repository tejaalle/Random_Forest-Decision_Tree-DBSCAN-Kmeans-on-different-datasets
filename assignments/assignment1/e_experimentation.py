import collections
import datetime
import itertools
from pathlib import Path
from typing import Union, Optional
from enum import Enum

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from assignments.assignment1.b_data_profile import *
from assignments.assignment1.c_data_cleaning import *
from assignments.assignment1.d_data_encoding import *
from assignments.assignment1.a_load_file import read_dataset
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


##############################################
# Example(s). Read the comments in the following method(s)
##############################################


def process_iris_dataset() -> pd.DataFrame:
    """
    In this example, I call the methods you should have implemented in the other files
    to read and preprocess the iris dataset. This dataset is simple, and only has 4 columns:
    three numeric and one categorical. Depending on what I want to do in the future, I may want
    to transform these columns in other things (for example, I could transform a numeric column
    into a categorical one by splitting the number into bins, similar to how a histogram creates bins
    to be shown as a bar chart).

    In my case, what I want to do is to *remove missing numbers*, replacing them with valid ones,
    and *delete outliers* rows altogether (I could have decided to do something else, and this decision
    will be on you depending on what you'll do with the data afterwords, e.g. what machine learning
    algorithm you'll use). I will also standardize the numeric columns, create a new column with the average
    distance between the three numeric column and convert the categorical column to a onehot-encoding format.

    :return: A dataframe with no missing values, no outliers and onehotencoded categorical columns
    """
    df = read_dataset(Path('..', '..', 'iris.csv'))
    numeric_columns = get_numeric_columns(df)
    categorical_columns = get_text_categorical_columns(df)

    for nc in numeric_columns:
        df = fix_outliers(df, nc)
        df = fix_nans(df, nc)
        df.loc[:, nc] = standardize_column(df.loc[:, nc])

    distances = pd.DataFrame()
    for nc_combination in list(itertools.combinations(numeric_columns, 2)):
        distances[str(nc_combination)] = calculate_numeric_distance(df.loc[:, nc_combination[0]],
                                                                    df.loc[:, nc_combination[1]],
                                                                    DistanceMetric.EUCLIDEAN).values
    df['numeric_mean'] = distances.mean(axis=1)
    for cc in categorical_columns:
        ohe = generate_one_hot_encoder(df.loc[:, cc])
        df = replace_with_one_hot_encoder(df, cc, ohe, list(ohe.get_feature_names()))
    return df


##############################################
# Implement all the below methods
# All methods should be dataset-independent, using only the methods done in the assignment
# so far and pandas/numpy/sklearn for the operations
##############################################
def process_iris_dataset_again() -> pd.DataFrame:
    df = read_dataset(Path('..', '..', 'iris.csv'))
    #collecting numeric and categorical columns in dataset
    numeric_columns = get_numeric_columns(df)
    categorical_columns = get_text_categorical_columns(df)
    #creating a new column based on the value of sepeal_length
    df["large_sepal_length"] = np.where(df["sepal_length"]>5,True,False)
    #normalizing the numeric columns
    for column in numeric_columns:
        df[column]= normalize_column(df[column])
    #label encoding the categorical columns
    for column in categorical_columns:
        df=replace_with_label_encoder(df,column,generate_label_encoder(df[column]))
    return df


def process_amazon_video_game_dataset():
    df = read_dataset(Path('..', '..', 'ratings_Video_Games.csv'))
    #taken from https://stackoverflow.com/questions/21787496/converting-epoch-time-with-milliseconds-to-datetime
    # df["time"]=df["time"].apply(lambda y:datetime.datetime.fromtimestamp(y / 1000.0).strftime('%Y-%m-%d %H:%M:%S.%f'))
    # df["time"]= pd.to_datetime(df["time"],unit="ms")
    #checking rating values are between 1 and 5
    #counting the number of user for each product
    user_count = df.groupby("asin")["user"].count()
    # finding mean of the ratings for each product
    review_avg = df.groupby("asin")["review"].mean()
    #creating columns count and review based on the above calculate values for each product
    df["count"]=df["asin"].apply(lambda y:user_count[y])
    df["review"]=df["asin"].apply(lambda y:review_avg[y])
    #dropping the user column
    df = df.drop("user",axis=1)
    return df


def process_amazon_video_game_dataset_again():
    df = read_dataset(Path('..', '..', 'ratings_Video_Games.csv'))
    # taken from https://stackoverflow.com/questions/21787496/converting-epoch-time-with-milliseconds-to-datetime
    # df["time"] = df["time"].apply(lambda y: datetime.datetime.fromtimestamp(y / 1000.0).strftime('%Y-%m-%d %H:%M:%S.%f'))
    # df["time"] = pd.to_datetime(df["time"], unit="ms")
    # checking rating values are between 1 and 5
    #finding mean, count, median, std of ratings of each user
    count_rating = df.groupby("user")["review"].count()
    avg_rating = df.groupby("user")["review"].mean()
    median_rating = df.groupby("user")["review"].median()
    std_rating = df.groupby("user")["review"].std()
    # creating column count,avg, median, std columns based on the above calcualted values for each user

    df["count"]=df["user"].apply(lambda y:count_rating[y])
    df["avg"]=df["user"].apply(lambda y:avg_rating[y])
    df["median"] = df["user"].apply(lambda y: median_rating[y])
    df["std"] = df["user"].apply(lambda y: std_rating[y] if(std_rating[y]==std_rating[y]) else 0)
    return df


def process_life_expectancy_dataset():
    df = read_dataset(Path('..', '..', 'life_expectancy_years.csv'))
    df1 = read_dataset(Path('..', '..', 'geography.csv'))
    numeric_columns = get_numeric_columns(df)
    categorical_columns = get_text_categorical_columns(df)
    for column in df.columns:
        df = fix_nans(df,column)
    for column in numeric_columns:
        df = fix_outliers(df,column)
    #making the all the years(219) as rows for each country with year as column
    df =df.melt(id_vars=["country"],var_name="Year",value_name="Value")
    #creating a new column country which is equal to name
    df1["country"]= df1["name"]
    #dropping name column as we have country column
    df1 = df1.drop("name",axis=1)
    #merging both the data frames
    df2 = pd.merge(left=df,right=df1,on="country",how="inner")
    df2 = df2.drop(["geo","eight_regions","six_regions","members_oecd_g77","Longitude","UN member since","World bank region","World bank, 4 income groups 2017","adjective","adjective_plural"],axis=1)
    #converting the latitude column based on its value if positive converted as north if negative converted a s south
    df2["Latitude"]=np.where(df2["Latitude"]>0,"north","south")
    df2= replace_with_label_encoder(df2, 'Latitude', generate_label_encoder(df2["Latitude"]))
    #generatinga and replacing continent column with one hot encoder
    ohe = generate_one_hot_encoder(df2["four_regions"])
    df2 = replace_with_one_hot_encoder(df2, 'four_regions', ohe, list(ohe.get_feature_names()))
    return df2


if __name__ == "__main__":
    assert process_iris_dataset() is not None
    assert process_iris_dataset_again() is not None
    assert process_amazon_video_game_dataset() is not None
    assert process_amazon_video_game_dataset_again() is not None
    assert process_life_expectancy_dataset() is not None
