import numpy as np
import argparse
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

def parse_argument():
    """
    Code for parsing arguments
    """
    parser = argparse.ArgumentParser(description='Parsing a file.')
    parser.add_argument('--train', nargs=1, required=True)
    parser.add_argument('--test', nargs=1, required=True)
    parser.add_argument('--numTrees', nargs=1, required=True)
    args = vars(parser.parse_args())
    return args


def adaboost(X, y, num_iter):
    """Given an numpy matrix X, a array y and num_iter return trees and weights 
   
    Input: X, y, num_iter
    Outputs: array of trees from DecisionTreeClassifier
             trees_weights array of floats
    Assumes y is in {-1, 1}^n
    """
    trees = []
    trees_alpha = []
    N = len(y)
    w = [1.0/N] * N
    for tree in range(num_iter):
        d_tree = DecisionTreeClassifier(max_depth=1)
        tree = d_tree.fit(X, y, sample_weight=w)
        pred = tree.predict(X)
        err_top = sum([1 * w[x] for x in range(N) if y[x] != pred[x]])
        err_bottom = sum(w)
        if err_top == 0 or err_bottom == 0:
            error = .0001
        else:
            error = sum([1 * w[x] for x in range(N) if y[x] != pred[x]]) / sum(w)
        alpha = np.log((1 - error) / error)
        alpha_list = [alpha] * N
        trees.append(tree)
        trees_alpha.append(alpha)
        w = [w[i] * np.exp(alpha_list[i] * 1) if y[i] != pred[i] else w[i] for i in range(N)]
    return trees, trees_alpha

def sign(x):
    if x < 0:
        return -1.
    return 1.

def adaboost_predict(X, trees, trees_alphas):
    """Given X, trees and weights predict Y
    assume Y in {-1, 1}^n
    """
    Yhat = []
    predlist = []
    for tree in trees:
        predlist.append(tree.predict(X))
    preds = np.array(predlist)

    for col in range(len(preds[0])):
        p = 0
        for row in range(len(preds)):
            p += trees_alphas[row] * preds[row][col]
        Yhat.append(sign(p))
    return Yhat

def parse_spambase_data(filename):
    """ Given a filename return X and Y numpy arrays
    X is of size number of rows x num_features
    Y is an array of size the number of rows
    Y is the last element of each row.
    """
    dataframe = pd.read_csv(filename, header = None)
    X = dataframe.iloc[:, :-1].as_matrix()
    Y = dataframe.iloc[:, -1].as_matrix()
    return X, Y

def new_label(Y):
    """ Transforms a vector od 0s and 1s in -1s and 1s.
    """
    return [-1 if y == 0. else 1 for y in Y]

def old_label(Y):
    return [0 if y == -1. else 1 for y in Y]

def accuracy(y, pred):
    return np.sum(y == pred) / float(len(y)) 

def main():
    """
    This code is called from the command line via
    python adaboost.py --train [path to filename] --test [path to filename] --numTrees
    """
    args = parse_argument()
    train_file = args['train'][0]
    test_file = args['test'][0]

    num_trees = int(args['numTrees'][0])
    print train_file, test_file ,num_trees

    test_df = pd.read_csv(test_file, header = None)

    train_X, train_Y = parse_spambase_data(train_file)
    test_X, test_Y = parse_spambase_data(test_file)

    trees, tree_alphas = adaboost(train_X, new_label(train_Y), num_trees)
    train_Yhat = old_label(adaboost_predict(train_X, trees, tree_alphas))
    test_Yhat = old_label(adaboost_predict(test_X, trees, tree_alphas))

    # COMMENT OUT IF DONT WANT TO  SAVE TO FILE#####################
    results = pd.DataFrame(test_Yhat)
    test_df = pd.concat([test_df, results], axis = 1)
    test_df.to_csv('predictions.txt', index = False, header = False)
    #################################################################

    acc = accuracy(train_Y, train_Yhat)
    acc_test = accuracy(test_Y, test_Yhat)

    print("Train Accuracy %.4f" % acc)
    print("Test Accuracy %.4f" % acc_test)

if __name__ == '__main__':
    main()