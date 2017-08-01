from nltk import ngrams
from collections import Counter
import re
from string import punctuation
import pandas as pd
# import math

whole_set = pd.read_csv('train.tsv', delimiter='\t', header= 0) # 17207
train_set= whole_set.sample(n = 16207)
whole_minus_train = whole_set.loc[~whole_set.index.isin(train_set.index)]
validation_set = whole_minus_train.sample(n = 500)
test_set = whole_minus_train.loc[~whole_minus_train.index.isin(validation_set.index)]


def word_list(text, contraction = False):
    punc_split = re.compile('[' + punctuation.replace("'", "") + ']')
    words = re.sub(punc_split, ' ', text).lower()
    if contraction:
        words = re.sub("[']", " '", words)
    words = words.split()
    return words

def makebigrams(text):
    return ngrams(text, 2)

def get_essays(df, score):
    """
    Subsets all essays in train_df with score equal to x
    """
    target_df = df[df["Score1"] == score]
    essay_col = target_df['EssayText']
    return essay_col

for set_id in range(1,11,1):
    """
    FURTHER IMPLEMENTATION REQUIRED:
    1. How to compare corpus distribution with individual distribution?
    2. Instead of a 'makebigrams' function, it will be making 'allgrams' function where we simultaneously
    compare results of unigram, bigram, trigram and quadgram distributions from the corpus and compare to the individual

    """

    # returns subset of training set that equals the set id
    question_set = train_set[train_set['EssaySet']==set_id]

    # returns the subset of the test set that equals the set id
    test_df = test_set[test_set['EssaySet']==set_id]

    # gets all the unique scores possible. some essays have 0-2 and some have 0-3
    scorelist = sorted(question_set['Score1'].unique())

    # creates a list of lists per different score
    # list_of_list_per_score = [get_essays(question_set, i) for i in scorelist]

    # gets a count of bigrams per essay in training set
    temp_df = question_set.copy()
    temp_df['wordlist'] = temp_df['EssayText'].map(lambda x : word_list(x))
    temp_df['bigramCounter'] = temp_df['wordlist'].map(lambda x : Counter(makebigrams(x)))
    grouped_essays = temp_df.groupby(['Score1'])

    # get distribution of bigrams per score of the set
    # score_counter_dicts[0] will correspond to distribution of score0

    score_counter_dicts = []
    for score in scorelist:
        set_of_this_score = grouped_essays.get_group(score)
        list_of_this_sets_bigrams = set_of_this_score['bigramCounter'].tolist()
        c = Counter()
        for dicts in list_of_this_sets_bigrams:
            c.update(dicts)
        score_counter_dicts.append(c)

    # makes bigram distribution per short answer
    test_df['wordlist'] = test_df['EssayText'].map(lambda x : word_list(x))
    test_df['bigramCounter'] = test_df['wordlist'].map(lambda x : Counter(makebigrams(x)))
    print ('complete')
    break