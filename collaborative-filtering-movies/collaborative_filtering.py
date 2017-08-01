import argparse
import re
import os
import csv
import math
import collections as coll
import pandas as pd
import numpy as np
import time

def parse_argument():
    """
    Code for parsing arguments
    """
    parser = argparse.ArgumentParser(description='Parsing a file.')
    parser.add_argument('--train', nargs=1, required=True)
    parser.add_argument('--test', nargs=1, required=True)
    args = vars(parser.parse_args())
    return args


def parse_file(filename):
    """
    Given a filename outputs user_ratings and movie_ratings dictionaries
    Input: filename
    Output: user_ratings, movie_ratings
        where:
            user_ratings[user_id] = {movie_id: rating}
            movie_ratings[movie_id] = {user_id: rating}
    """
    data = pd.read_csv(filename, header = None, names = ['MovieId', 'CustomerId', 'Rating'])
    user_ratings = {user_id: df.set_index('MovieId')['Rating'].to_dict() for user_id, df in data.groupby('CustomerId')}
    movie_ratings = {movie_id: df.set_index('CustomerId')['Rating'].to_dict() for movie_id, df in data.groupby('MovieId')}
    return user_ratings, movie_ratings


def compute_average_user_ratings(user_ratings):
    """ Given a the user_rating dict compute average user ratings
    Input: user_ratings (dictionary of user, movies, ratings)
    Output: ave_ratings (dictionary of user and ave_ratings)
    """
    avg_ratings = {user_id: np.mean(ratings.values()) for user_id, ratings in user_ratings.items()}
    return avg_ratings


def compute_user_similarity(d1, d2, ave_rat1, ave_rat2):
    """ Computes similarity between two users
        Input: d1, d2, (dictionary of user ratings per user)
            ave_rat1, ave_rat2 average rating per user (float)
        Ouput: user similarity (float)
    """
    keys_intersect = set(d1.keys()) & set(d2.keys())
    if len(keys_intersect) == 0:
        return 0.0
    top = [float(d1[key] - ave_rat1) * float(d2[key] - ave_rat2) for key in keys_intersect]
    top_sum = sum(top)
    bottom1 = [float(d1[key] - ave_rat1)**2 for key in keys_intersect]
    bottom2 = [float(d2[key] - ave_rat2)**2 for key in keys_intersect]
    bottom_sum = math.sqrt(sum(bottom1) * sum(bottom2))
    if top_sum == 0 or bottom_sum == 0:
        return 0
    similarity = top_sum/bottom_sum
    return similarity

def main():
    """
    This function is called from the command line via
    python cf.py --train [path to filename] --test [path to filename]
    """
    args = parse_argument()
    train_file = args['train'][0]
    test_file = args['test'][0]
    # print train_file, test_file  # prints the filenames

    # convert test dataframe to a NumPy matrix for faster calculations
    test_df = pd.read_csv(test_file, header = None,names = ['MovieId', 'CustomerId', 'Rating'])
    test_matrix = test_df.as_matrix()

    # returns (user_ratings, movie_ratings) tuple for train + test set
    train_user_movie = parse_file(train_file)

    # finds the average rating per user in train
    avg_rat_train = compute_average_user_ratings(train_user_movie[0])

    user_similarity = {}
    all_preds  = []
    # count = 1
    for row in range(test_matrix.shape[0]):
        MovieId = test_matrix[row][0]
        test_user = test_matrix[row][1]

        # find user's average rating
        user_avg = avg_rat_train[test_user]

        # find all users who have watched this movie
        watched_user_dict = train_user_movie[1][MovieId]

        bottom_sum_pred = 0
        right_sum = 0
        for train_user in watched_user_dict:
            if train_user > test_user:
                user1_user2_tup = (test_user, train_user)
            else:
                user1_user2_tup = (train_user, test_user)

            user_d1 = train_user_movie[0][test_user]
            user_d2 = train_user_movie[0][train_user]

            ave_rat1 = avg_rat_train[test_user]
            ave_rat2 = avg_rat_train[train_user]

            if user1_user2_tup not in user_similarity:
                similarity = compute_user_similarity(user_d1, user_d2, ave_rat1, ave_rat2)
                user_similarity[user1_user2_tup] = similarity

            sim_score = user_similarity[user1_user2_tup]
            bottom_sum_pred += abs(sim_score)
            right_sum += sim_score * float(train_user_movie[0][train_user][MovieId] - avg_rat_train[train_user])

        if bottom_sum_pred == 0 or right_sum == 0:
            pred = user_avg
        else:
            pred = user_avg + (right_sum / bottom_sum_pred)
        all_preds.append(pred)
        # print count
        # count +=1

    test_df['Predictions'] = pd.Series(all_preds).values
    test_df.to_csv('predictions.txt', index = False, header = False)
    observed = test_matrix[:,2]
    rmse = math.sqrt(sum((observed-all_preds)**2/test_matrix.shape[0]))
    mae = sum((abs(observed-all_preds))/test_matrix.shape[0])
    print "RMSE %s" %round(rmse, 4)
    print "MAE %s" %round(mae, 4)

if __name__ == '__main__':
    main()