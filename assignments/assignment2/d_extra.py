from pathlib import Path
from typing import List, Optional, Dict
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, MeanShift,AffinityPropagation
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from assignments.assignment1.b_data_profile import get_numeric_columns,get_binary_columns,get_text_categorical_columns
from assignments.assignment1.d_data_encoding import generate_one_hot_encoder,replace_with_one_hot_encoder,generate_label_encoder,replace_with_label_encoder,fix_nans,fix_outliers
from assignments.assignment2.b_regression import simple_random_forest_regressor,decision_tree_regressor
from assignments.assignment2.a_classification import simple_random_forest_classifier,decision_tree_classifier

"""
COMPETITION EXTRA POINTS!!
The below method should:
1. Handle any dataset (if you think worthwhile, you should do some preprocessing)
2. Generate a model based on the label_column and return the one with best score/accuracy

The label_column may indicate categorical column as label, numerical column as label or it can also be None
If categorical, run through these ML classifiers and return the one with highest accuracy: 
    DecisionTree, RandomForestClassifier, KNeighborsClassifier or a NaiveBayes
If numerical, run through these ML regressors and return the one with highest R^2: 
    DecisionTree, RandomForestRegressor, KNeighborsRegressor or a Gaussian NaiveBayes
If None, run through at least 4 of the ML clustering algorithms in https://scikit-learn.org/stable/modules/clustering.html
and return the one with highest silhouette (https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html)

Optimize all parameters for the given score above. Feel free to choose any method you wish to optimize these parameters.
Write as a comment why and how you decide your model_type/parameter combination.
This method should not take more than 10 minutes to finish in a desktop Ryzen5 (or Core i5) CPU (no gpu acceleration).  

We will run your code with a separate dataset unknown to you. We will call the method more then once with said dataset, measuring
all scores listed above. The 5 best students of each score type will receive up to 5 points (first place->5 points, second->4, and so on).
Finally, the top 5 students overall (with most points in the end) will be awarded a prize!
"""

def NaiveBayes(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    chunks = len(str(X_train.shape[0]))
    # If the size of training data is grater than 10000 than we will divide them into chunks otherwise we will do it normally
    model = GaussianNB()
    if (chunks > 3):
        first = 0
        inc = X_train.shape[0] // (chunks * 40)
        second = inc
        for i in range(chunks * 40):
            model = model.fit(X_train[first:second], y_train[first:second])
            first = second
            second = second + inc
    else:
        model = model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    accuracy = model.score(X_test, y_test)
    return dict(model=model, accuracy=accuracy, test_prediction=y_predict)
def generate_model(df: pd.DataFrame, label_column: Optional[str]) -> Dict:
    numeric = get_numeric_columns(df)
    categorical = get_text_categorical_columns(df)
    #converting categorical columns into labels depending on the number
    for column in categorical:
        #if the the values in columns are less than 200 and not label column(for classification) then one hot encoding is used other wise label encoder is used
        if(len(df[column].unique())<200 and column!=label_column ):
            ohe = generate_one_hot_encoder(df[column])
            df = replace_with_one_hot_encoder(df, column, ohe, list(ohe.get_feature_names()))
        else:
            le = generate_label_encoder(df[column])
            df = replace_with_label_encoder(df, column=column, le=le)
    #removing nans and fixing outliners from the numeric columns, I don't want to normalize becayse I see many models provide more accuracy on not normalized data.
    for column in numeric:
        df =fix_nans(df,column)
        df = fix_outliers(df,column)
    if label_column is None:
        #number of features are used as min_samples and 1.4 epsilon is used which may fit all datasets
        dbscan = DBSCAN(eps=1.4, min_samples=len(df.columns))
        db_clusters =dbscan.fit(df)
        db_score = metrics.silhouette_score(df, dbscan.labels_, sample_size=1000)
        #bandwidth 2 is considered after some research hope  it satisfies all datasets
        meanshift = MeanShift(bandwidth=2)
        meanshift_clusters =meanshift.fit(df)
        meanshift_score = metrics.silhouette_score(df,meanshift.labels_,sample_size=1000)
        #damping is taken 0.9 after running on some datasets
        Affinity = AffinityPropagation(damping=0.9)
        affinity_clusters = Affinity.fit(df)
        affinity_score = metrics.silhouette_score(df,Affinity.labels_,sample_size=1000)
        max_score = max(db_score,affinity_score,meanshift_score)
        if(max_score==db_score):
            return dict(model=dbscan, score=db_score, clusters=db_clusters)
        elif(max_score==affinity_score):
            return dict(model=Affinity, score=affinity_score, clusters=affinity_clusters)
        else:
            return dict(model=meanshift, score=meanshift_score, clusters=meanshift_clusters)

    elif label_column in numeric:
        X = df.drop(label_column,axis=1)
        Y = df[label_column]
        random = simple_random_forest_regressor(X=X,y=Y)
        decision = decision_tree_regressor(X,Y)
        if(random["score"]>decision["score"]):
            return dict(model=random["model"], accuracy=random["score"], test_prediction=random["test_prediction"])
        else:
            return dict(model=decision["model"], accuracy=decision["score"], test_prediction=decision["test_prediction"])
    else:
        X = df.drop(label_column,axis=1)
        Y = df[label_column]
        random = simple_random_forest_classifier(X,Y)
        decision = decision_tree_classifier(X,Y)
        Naive = NaiveBayes(X,Y)
        max_accuracy = max(Naive["accuracy"],random["accuracy"],decision["accuracy"])
        if(max_accuracy==random["accuracy"]):
            return random
        elif(max_accuracy==decision["accuracy"]):
            return decision
        else:
            return Naive










    return dict(model=None, final_score=None)
