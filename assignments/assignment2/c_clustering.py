from pathlib import Path
from typing import List, Dict
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans, DBSCAN
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from datetime import datetime
from assignments.assignment1.a_load_file import read_dataset
from assignments.assignment1.d_data_encoding import fix_outliers, fix_nans, normalize_column, \
    generate_one_hot_encoder, replace_with_one_hot_encoder, generate_label_encoder, replace_with_label_encoder
from assignments.assignment1.e_experimentation import process_iris_dataset, process_amazon_video_game_dataset,process_iris_dataset_again,process_life_expectancy_dataset, process_amazon_video_game_dataset_again

"""
Clustering is a non-supervised form of machine learning. It uses unlabeled data
 through a given method, returns the similarity/dissimilarity between rows of the data.
 See https://scikit-learn.org/stable/modules/clustering.html for an overview of methods in sklearn.
"""


##############################################
# Example(s). Read the comments in the following method(s)
##############################################
def simple_k_means(X: pd.DataFrame, n_clusters=3, score_metric='euclidean') -> Dict:
    model = KMeans(n_clusters=n_clusters)
    clusters = model.fit_transform(X)
    # There are many methods of deciding a score of a cluster model. Here is one example:
    score = metrics.silhouette_score(X, model.labels_, metric=score_metric)
    return dict(model=model, score=score, clusters=clusters)


def iris_clusters() -> Dict:
    """
    Let's use the iris dataset and clusterise it:
    """
    df = pd.read_csv(Path('..', '..', 'iris.csv'))
    for c in list(df.columns):
        df = fix_outliers(df, c)
        df = fix_nans(df, c)
        df[c] = normalize_column(df[c])

    # Let's generate the clusters considering only the numeric columns first
    no_species_column = simple_k_means(df.iloc[:, :4])

    ohe = generate_one_hot_encoder(df['species'])
    df_ohe = replace_with_one_hot_encoder(df, 'species', ohe, list(ohe.get_feature_names()))

    # Notice that here I have binary columns, but I am using euclidean distance to do the clustering AND score evaluation
    # This is pretty bad
    no_binary_distance_clusters = simple_k_means(df_ohe)
    # Finally, lets use just a label encoder for the species.
    # It is still bad to change the labels to numbers directly because the distances between them does not make sense
    le = generate_label_encoder(df['species'])
    df_le = replace_with_label_encoder(df, 'species', le)
    labeled_encoded_clusters = simple_k_means(df_le)

    # See the result for yourself:
    print(no_species_column['score'], no_binary_distance_clusters['score'], labeled_encoded_clusters['score'])
    ret = no_species_column
    if no_binary_distance_clusters['score'] > ret['score']:
        ret = no_binary_distance_clusters
    if labeled_encoded_clusters['score'] > ret['score']:
        ret = labeled_encoded_clusters
    return ret


##############################################
# Implement all the below methods
# Don't install any other python package other than provided by python or in requirements.txt
##############################################
def custom_clustering(X: pd.DataFrame,eps,samples) -> Dict:
    """
    As you saw before, it is much harder to apply the right distance metrics. Take a look at:
    https://scikit-learn.org/stable/modules/clustering.html
    and check the metric used for each implementation. You will notice that suppositions were made,
    which makes harder to apply these clustering algorithms as-is due to the metrics used.
    Also go into each and notice that some of them there is a way to choose a distance/similarity/affinity metric.
    You don't need to check how each technique is implemented (code/math), but do use the information from the clustering
    lecture and check the parameters of the method (especially if there is any distance metric available among them).
    Chose one of them which is, in your opinion, generic enough, and justify your choice with a comment in the code (1 sentence).
    The return of this method should be the model, a score (e.g. silhouette
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html) and the result of clustering the
    input dataset.
    """
    # I have choosen DBSCAN algorithm because it is best in handling noisy data and high density data. It is very  scalable and suitable for large datasets like we have. We dont have to provide the number of groups in before hand. Since We have datasets with more noise I prefer to use this dbsacn
    dbscan = DBSCAN(eps=eps,min_samples=samples)
    clusters=dbscan.fit(X)
    #I have taken only 1000 samples to check the score because check 1000 samples saves times it's score is approximately equal to to the total samples score
    score = metrics.silhouette_score(X, dbscan.labels_,sample_size=1000)
    return dict(model=dbscan, score=score, clusters=clusters)


def cluster_iris_dataset_again() -> Dict:
    """
    Run the result of the process iris again task of e_experimentation through the custom_clustering and discuss (3 sentences)
    the result (clusters and score) and also say any limitations (e.g. problems with metrics) that you find.
    We are not looking for an exact answer, we want to know if you really understand your choice and the results of custom_clustering.
    Once again, don't worry about the clustering technique implementation, but do analyse the data/result and check if the clusters makes sense.
    """
    df = process_iris_dataset_again()
    #There are certain limitation in using DBSCAN on this dataset if we miss sending the larger_sepal_length column. This algorithm classifies all into single group
    #This is because of all the values are normalized before in the e-experimentation. All the values lies between 0 and 1. so distance between them is not the real distance.
    #And I have tried different epsilon values. Epsilon values above 1 is grouping all the elements into single group as we increasing the radius all the points are being recognized as
    #core points. So I took epsilon as 1. Taking more number of min-samples does not have any effect as we have radius 1 and doesnot contain more samples so increasing does not matter. Taking less samples increases the groups but I didn't find more possible groups then four
    #so I went with 4 as min samples., I always start the minsamples from number of dimensions
    dbscan = custom_clustering(df,1,4)
    return dict(model=dbscan["model"], score=dbscan["score"], clusters=dbscan["clusters"])


def cluster_amazon_video_game() -> Dict:
    """
    Run the result of the process amazon_video_game task of e_experimentation through the custom_clustering and discuss (max 3 sentences)
    the result (clusters and score) and also say any limitations (e.g. problems with metrics) that you find.
    We are not looking for an exact answer, we want to know if you really understand your choice and the results of custom_clustering.
    Once again, don't worry about the clustering technique implementation, but do analyse the data/result and check if the clusters makes sense.
    """
    df = process_amazon_video_game_dataset()
    #I have changed in e_experimentation to leave the time in millisecond because data cannot be used to find the distance only milliseconds will be useful
    le = generate_label_encoder(df["asin"])
    df = replace_with_label_encoder(df, column='asin', le=le)
    #Since we have grouped based on the asin we will get many groups as many as number of products. so I am thinking to group products which are closely
    #related so I have taken 0.5 as epsilon value with samples as 3, I always start the minsamples from number of dimensions
    #Limitation is I am geeting the negative score for this huge dataset. I have tried many epsilon and min_sample values But there is no effect
    dbscan = custom_clustering(df,0.5,3)
    return dict(model=dbscan["model"], score=dbscan["score"], clusters=dbscan["clusters"])


def cluster_amazon_video_game_again() -> Dict:
    """
    Run the result of the process amazon_video_game_again task of e_experimentation through the custom_clustering and discuss (max 3 sentences)
    the result (clusters and score) and also say any limitations (e.g. problems with metrics) that you find.
    We are not looking for an exact answer, we want to know if you really understand your choice and the results of custom_clustering.
    Once again, don't worry about the clustering technique implementation, but do analyse the data/result and check if the clusters makes sense.
    """
    df = process_amazon_video_game_dataset_again()
    # I have changed in e_experimentation to leave the time in millisecond because data cannot be used to find the distance only milliseconds will be useful
    le = generate_label_encoder(df["asin"])
    df = replace_with_label_encoder(df, column='asin', le=le)
    #here we have already made different columns like count, avg, median,std based on the user so we are here grouping users who have reviewed different products.
    #there is problem here with label encoding beacause user at first row and user at last row are as different as user at first row and second row but due to label encoding
    #the user at first row and last row seems to have more difference
    #As we already have details of the user in the count, avg, median, std columns and to avoid the above mentioned problem I have removing the user column.
    #on cosidering the epsilon value I started with 1 increasing the epsilon value is increasing the negative score seven on decreainge the epsilon value it showed same result so I have used 1 and
    #after experementation of min samples I used 7, I always start the minsamples from number of dimensions
    #Due to size of the amazon data size It is grouping into many number of groups But it not clear whether the formed groups are realated It comes under limitation
    df = df.drop("user",axis=1)
    dbscan = custom_clustering(df, 1, 7)
    return dict(model=dbscan["model"], score=dbscan["score"], clusters=dbscan["clusters"])


def cluster_life_expectancy() -> Dict:
    """
    Run the result of the process life_expectancy task of e_experimentation through the custom_clustering and discuss (max 3 sentences)
    the result (clusters and score) and also say any limitations (e.g. problems with metrics) that you find.
    We are not looking for an exact answer, we want to know if you really understand your choice and the results of custom_clustering.
    Once again, don't worry about the clustering technique implementation, but do analyse the data/result and check if the clusters makes sense.
    """
    df = process_life_expectancy_dataset()
    #I have used one hot encoding in country column so that It does not cause issue in calculating distnce between countries in the country column
    ohe = generate_one_hot_encoder(df.loc[:, "country"])
    df = replace_with_one_hot_encoder(df, "country", ohe, list(ohe.get_feature_names()))
    #I have started default value of epsilon 0.5 but it is unable to group it with it beacuase no cuntry was so ckose so I moved up to 1 It is making 1 group with some countries thrown into noise
    #category. So I moved up to 1.5 which is showing some reasonable grouping of countries based on the values and years. But I still want to group them and so increased to 2. For min values, i started considering from default value from 5
    #I directly increased it to 13 But its showing so weird results and then decreased to 7 which is working fine.I always start the minsamples from number of dimensions
    dbscan = custom_clustering(df,2,7)
    return dict(model=dbscan["model"], score=dbscan["score"], clusters=dbscan["clusters"])


if __name__ == "__main__":
    iris_clusters()
    assert cluster_iris_dataset_again() is not None
    assert cluster_amazon_video_game() is not None
    assert cluster_amazon_video_game_again() is not None
    assert cluster_life_expectancy() is not None
