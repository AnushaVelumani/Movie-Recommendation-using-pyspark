# Movie Recommender System using Apache Spark
# Abstract
The goal  of this project is to develop a collaborative filtering model that predicts the rating of a specific user rating given history of other ratings.
The performance of the results are evaluated using Root-Mean-Square-Error (RMSE)

# Dataset
The dataset consists of two files, train and test. Train dataset has 85724 ratings whereas the test dataset consists of 2154 ratings. Both the datasets are in the .dat format.

# Description
Collaborative Filtering (CF) systems measure similarity of users by their preferences and to measure similarity of items by users who like them. For this CF systems item profiles and user profiles are extracted and then similarity of rows and columns in the Utility Matrix are computed.

Here in this project, Alternating Least Squares (ALS) algorithm is used and its parameters are tuned to find the least RMSE.


