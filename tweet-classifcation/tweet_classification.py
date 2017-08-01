import pandas as pd
import re
from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis #, QuadraticDiscriminantAnalysis
# from sklearn.cross_validation import KFold
# from sklearn.linear_model import LogisticRegression
from collections import Counter
import sys

trainfile = sys.argv[1]
testfile = sys.argv[2]

f_train = open(trainfile, 'r')
train_lines = f_train.readlines()
f_train.close()

f_test = open(testfile, 'r')
test_lines = f_test.readlines()
f_test.close()

features = ['i', 'the', 'and', 'to', 'a', 'of', 'that', 'in', 'it', 'my', 'is',
                  'you', 'was', 'for', 'have','with', 'he', 'me', 'on', 'but', '.',',','!']

def clean_tweet(s):
    s = re.sub('\d', '',s)
    s = re.sub('([=/?#.,!:\'@-])', r' \1 ', s)
    s = re.sub('\s{2,}', ' ', s)
    s = s.lower().replace('"','').strip().split(' ')
    return s

# Count occurrences of terms in tweet
def count_word(tweet, term):
    tweet = ' '.join(tweet)
    sum = tweet.count(' '+ term +' ')
    return sum

def fill_rates(feature_list, dataframe):
    for term in feature_list:
        dataframe[term] = dataframe['tweets'].map(lambda x: float(count_word(x, term))/len(x))

def LDA(train_x, train_y, test_x):  # , Y_test):
    algo = LinearDiscriminantAnalysis()
    algo.fit(train_x, train_y)
    hyp = algo.predict(test_x)
    return hyp

# def QDA(train_x, train_y, test_x):
#     algo = QuadraticDiscriminantAnalysis()
#     algo.fit(train_x, train_y)
#     hyp = algo.predict(test_x)
#     return hyp
#
# def logistic(train_x, train_y, test_x):
#     log_reg = LogisticRegression(penalty = 'l1')
#     log_reg.fit(train_x, train_y)
#     hyp = log_reg.predict(test_x)
#     return hyp

# def kfold(dataframe, method):
#     misclass_rates = []
#     kf = KFold(len(dataframe), n_folds=10, shuffle=True)
#     for train_index, test_index in kf:
#         train_x = dataframe.ix[train_index, 2:]
#         train_y = dataframe.ix[train_index, 0]
#         test_x = dataframe.ix[test_index, 2:]
#         test_y = dataframe.ix[test_index, 0]
#         predicted = method(train_x, train_y, test_x)
#         misclass_rates.append(1 - metrics.accuracy_score(test_y, predicted))
#     return misclass_rates

# Bag of words
def most_used_words(dataframe):
    grouped = dataframe.groupby('sentiment')
    neutrals = grouped.get_group('neutral')
    negatives = grouped.get_group('negative')
    positives = grouped.get_group('positive')

    sentiment_list = [neutrals, negatives, positives]

    counters = []
    for sentiment in sentiment_list:
        sentiment_flat= sentiment['tweets'].tolist()
        counters.append(Counter(word for tweet in sentiment_flat for word in tweet).most_common(230))

    neutral_feats = [tuple[0] for tuple in counters[0]]
    negative_feats = [tuple[0] for tuple in counters[1]]
    positive_feats = [tuple[0] for tuple in counters[2]]

    most_frequent = neutral_feats + negative_feats + positive_feats + features
    most_frequent = list(set(most_frequent))
    #most_frequent = [word for word in most_frequent if len(word) > 1]
    remove = ['//']
    for phrase in remove:
        most_frequent = [word for word in most_frequent if not phrase in word]
    return most_frequent

# Returns list containing list of sentiments and list of tweets
def read_tweets(file_lines):
    sentiment_str = []
    tweets_str = []
    for line in file_lines:
        clean = line.rstrip().replace(", ", "_", 1).split("_")
        sentiment_str.append(clean[0])
        tweets_str.append(clean[1])
    return [sentiment_str, tweets_str]

# Make training set dataframe
read_train = read_tweets(train_lines)
train_df = pd.DataFrame()
train_df['sentiment'] = read_train[0]
train_df['tweets'] = read_train[1]

# Make test set dataframe
read_test = read_tweets(test_lines)
test_df = pd.DataFrame()
test_df['sentiment'] = read_test[0]
test_df['tweets'] = read_test[1]

# Clean and split each string
train_df['tweets'] = train_df['tweets'].map(lambda x: clean_tweet(x))
test_df['tweets'] = test_df['tweets'].map(lambda x: clean_tweet(x))

# Fill all features
fill_rates(most_used_words(train_df), train_df)
fill_rates(most_used_words(train_df), test_df)

train_x = train_df.ix[0:,2:]
train_y = train_df.ix[0:,0]
test_x = test_df.ix[0:,2:]
test_y = test_df.ix[0:,0]

predicted = LDA(train_x, train_y, test_x)
LDA_misclass_rate = 1 - metrics.accuracy_score(test_y, predicted)

print 'Misclassifcation Rate = %s' %LDA_misclass_rate