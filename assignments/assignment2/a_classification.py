from pathlib import Path
from typing import List, Dict
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier


from assignments.assignment1.a_load_file import read_dataset
from assignments.assignment1.d_data_encoding import generate_label_encoder, replace_with_label_encoder, fix_outliers, fix_nans, normalize_column,generate_one_hot_encoder,replace_with_one_hot_encoder
from assignments.assignment1.e_experimentation import process_iris_dataset, process_amazon_video_game_dataset_again, process_iris_dataset_again, process_life_expectancy_dataset

"""
Classification is a supervised form of machine learning. It uses labeled data, which is data with an expected
result available, and uses it to train a machine learning model to predict the said result. Classification
focuses in results of the categorical type.
"""


##############################################
# Example(s). Read the comments in the following method(s)
##############################################
def simple_random_forest_classifier(X: pd.DataFrame, y: pd.Series) -> Dict:
    """
    Simple method to create and train a random forest classifier
    https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)

    chunks = len(str(X_train.shape[0]))
    # If the size of training data is grater than 10000 than we will divide them into chunks otherwise we will do it normally
    #I have taken N_estimators = 4 that means it will construct 4 trees and take the output whhich maximum trees predict thus reducing the bias
    #To maintain time with out effecting the accuracy I have taken the max_depth=7 and n_estimaters =4, Increasing the depth and estimaters increases the accuracy but take time constraint I have done this
    model = RandomForestClassifier(n_estimators=4 , max_depth=8)
    if (chunks > 3):
        first = 0
        inc = X_train.shape[0] // (chunks * 40)
        second = inc
        for i in range(chunks * 40):
            model = model.fit(X_train[first:second], y_train[first:second])
            first = second
            second = second + inc
    else:
        model = RandomForestClassifier()
        model = model.fit(X_train, y_train)
    y_predict = model.predict(X_test)  # Use this line to get the prediction from the model
    accuracy = model.score(X_test, y_test)
    return dict(model=model, accuracy=accuracy, test_prediction=y_predict)


def simple_random_forest_on_iris() -> Dict:
    """
    Here I will run a classification on the iris dataset with random forest
    """
    df = pd.read_csv(Path('..', '..', 'iris.csv'))
    X, y = df.iloc[:, :4], df.iloc[:, 4]
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    return simple_random_forest_classifier(X, y_encoded)


def reusing_code_random_forest_on_iris() -> Dict:
    """
    Again I will run a classification on the iris dataset, but reusing
    the existing code from assignment1. Use this to check how different the results are (score and
    predictions).
    """
    df = read_dataset(Path('..', '..', 'iris.csv'))
    for c in list(df.columns):
        # Notice that I am now passing though all columns.
        # If your code does not handle normalizing categorical columns, do so now (just return the unchanged column)
        df = fix_outliers(df, c)
        df = fix_nans(df, c)
        df[c] = normalize_column(df[c])

    X, y = df.iloc[:, :4], df.iloc[:, 4]
    le = generate_label_encoder(y)
    # Be careful to return a copy of the input with the changes, instead of changing inplace the inputs here!
    y_encoded = replace_with_label_encoder(y.to_frame(), column='species', le=le)
    return simple_random_forest_classifier(X, y_encoded['species'])



##############################################
# Implement all the below methods
# Don't install any other python package other than provided by python or in requirements.txt
##############################################
def random_forest_iris_dataset_again() -> Dict:
    """
    Run the result of the process iris again task of e_experimentation and discuss (1 sentence)
    the differences from the above results. Use the same random forest method.
    Feel free to change your e_experimentation code (changes there will not be considered for grading
    purposes) to optimise the model (e.g. score, parameters, etc).
    """
    df = process_iris_dataset_again()
    #While comparing with above ones sometimes the accuracy is increasing and sometimes it remains equal and sometimes it is less than the above ones. It is fluctuating
    #Most of the times this accuracy is greater than the above ones.
    X, y = df.iloc[:, [0,1,2,3,5]], df["species"]
    return simple_random_forest_classifier(X, y)



def decision_tree_classifier(X: pd.DataFrame, y: pd.Series) -> Dict:
    """
    Reimplement the method "simple_random_forest_classifier" but using the technique we saw in class: decision trees
    (you can use sklearn to help you).
    Optional: also optimise the parameters of the model to maximise accuracy
    :param X: Input dataframe
    :param y: Label data
    :return: model, accuracy and prediction of the test set
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)

    #If the size of training data is greater than 10000 than I will divide them into chunks otherwise I will do it normally
    chunks = len(str(X_train.shape[0]))
    if(chunks>3):
        #To decrease the execution time without much effecting the accuracy, I Have taken depth =10 which I think would be enough to maintain the accuracy
        #Increasing depth will definitely increase the accuracy but taking time constarint I have taken depth 10
        model = DecisionTreeClassifier(max_depth=10)
        first = 0
        inc = X_train.shape[0] // (chunks * 40)
        second = inc
        for i in range(chunks*40):
            model = model.fit(X_train[first:second],y_train[first:second])
            first = second
            second = second+inc
    else:
        model = DecisionTreeClassifier()
        model = model.fit(X_train,y_train)

    y_predict = model.predict(X_test)  # Use this line to get the prediction from the model
    accuracy = model.score(X_test, y_test)
    return dict(model=model, accuracy=accuracy, test_prediction=y_predict)


def train_iris_dataset_again() -> Dict:
    """
    Run the result of the iris dataset again task of e_experimentation using the
    decision_tree classifier AND random_forest classifier. Return the one with highest score.
    Discuss (1 sentence) what you found different between the two models and scores.
    Feel free to change your e_experimentation code (changes there will not be considered for grading
    purposes) to optimise the model (e.g. score, parameters, etc).
    """
    df = process_iris_dataset_again()
    X, y = df.iloc[:, :4], df["species"]
    decision = decision_tree_classifier(X,y)
    random = simple_random_forest_classifier(X,y)
    #After running many times, both models gives good accuaracy but RandomForest Classifier has upper hand, Accuracy sometimes reaches 1.

    if(decision["accuracy"]>random["accuracy"]):
        return dict(model=decision["model"], accuracy=decision["accuracy"], test_prediction=decision["test_prediction"])
    else:
        return dict(model=random["model"], accuracy=random["accuracy"], test_prediction=random["test_prediction"])



def train_amazon_video_game_again() -> Dict:
    """
    Run the result of the amazon dataset again task of e_experimentation using the
    decision tree classifier AND random_forest classifier. Return the one with highest score.
    The Label column is the user column. Choose what you wish to do with the time column (drop, convert, etc)
    Discuss (1 sentence) what you found different between the results.
    In one sentence, why is the score worse than the iris score (or why is it not worse) in your opinion?
    Feel free to change your e_experimentation code (changes there will not be considered for grading
    purposes) to optimise the model (e.g. score, parameters, etc).
    """

    df= process_amazon_video_game_dataset_again()
    X, y = df.iloc[:, 2:6], df["user"]
    le = generate_label_encoder(y)
    #I have removed the standard deviation column since most of the values in the column is nan which is of no using while predictung
    #I have removed asin(product) column because while predicting the user, product will be of no use and we already have more data realted to user in review, count, avg, median
    #adding product column will cause bias.
    #The accuracy of both are coming similar but less than the previous ones on iris data set. But random forest has the upper hand more number of times.
    y_encoded = replace_with_label_encoder(y.to_frame(), column='user', le=le)
    random = simple_random_forest_classifier(X,y_encoded['user'])
    decision = decision_tree_classifier(X, y_encoded['user'])
    if (decision["accuracy"] > random["accuracy"]):
        return dict(model=decision["model"], accuracy=decision["accuracy"], test_prediction=decision["test_prediction"])
    else:
        return dict(model=random["model"], accuracy=random["accuracy"], test_prediction=random["test_prediction"])


def train_life_expectancy() -> Dict:
    """
    Do the same as the previous task with the result of the life expectancy task of e_experimentation.
    The label column is the column which has north/south. Remember to convert drop columns you think are useless for
    the machine learning (say why you think so) and convert the remaining categorical columns with one_hot_encoding.
    (check the c_regression examples to see example on how to do this one hot encoding)
    Feel free to change your e_experimentation code (changes there will not be considered for grading
    purposes) to optimise the model (e.g. score, parameters, etc).
    """
    df= process_life_expectancy_dataset()
    X, y = df.iloc[:, :7], df["Latitude"]
    ohe = generate_one_hot_encoder(X.loc[:, "country"])
    X = replace_with_one_hot_encoder(X, "country", ohe, list(ohe.get_feature_names()))
    #I found all the columns are useful and did not drop any of them
    le = generate_label_encoder(y)
    y_encoded = replace_with_label_encoder(y.to_frame(), column='Latitude', le=le)
    decision = decision_tree_classifier(X, y_encoded["Latitude"])
    random = simple_random_forest_classifier(X,y_encoded["Latitude"])
    if (decision["accuracy"] > random["accuracy"]):
        return dict(model=decision["model"], accuracy=decision["accuracy"], test_prediction=decision["test_prediction"])
    else:
        return dict(model=random["model"], accuracy=random["accuracy"], test_prediction=random["test_prediction"])


def your_choice() -> Dict:
    """
    Now choose one of the datasets included in the assignment1 (the raw one, before anything done to them)
    and decide for yourself a set of instructions to be done (similar to the e_experimentation tasks).
    Specify your goal (e.g. analyse the reviews of the amazon dataset), say what you did to try to achieve the goal
    and use one (or both) of the models above to help you answer that. Remember that these models are classification
    models, therefore it is useful only for categorical labels.
    We will not grade your result itself, but your decision-making and suppositions given the goal you decided.
    Use this as a small exercise of what you will do in the project.
    """
    df = read_dataset(Path('..', '..', 'iris.csv'))
    #My goal to find the species of each flower based on remaining features
    #I want to see how the models work for the not normalized data and finally return the one with more accuracy.
    for c in list(df.columns):
        df = fix_outliers(df, c)
        df = fix_nans(df, c)
    le = generate_label_encoder(df["species"])
    df = replace_with_label_encoder(df, column='species', le=le)
    X, y = df.iloc[:, :4], df.iloc[:, 4]
    decision = decision_tree_classifier(X, y)
    random = simple_random_forest_classifier(X, y)
    if (decision["accuracy"] > random["accuracy"]):
        return dict(model=decision["model"], accuracy=decision["accuracy"], test_prediction=decision["test_prediction"])
    else:
        return dict(model=random["model"], accuracy=random["accuracy"], test_prediction=random["test_prediction"])


if __name__ == "__main__":

    assert simple_random_forest_on_iris() is not None
    assert reusing_code_random_forest_on_iris() is not None
    assert random_forest_iris_dataset_again() is not None
    assert train_iris_dataset_again() is not None
    assert train_amazon_video_game_again() is not None
    assert train_life_expectancy() is not None
    assert your_choice() is not None

