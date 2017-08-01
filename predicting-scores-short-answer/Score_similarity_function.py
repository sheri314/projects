from string import punctuation
import re
from nltk.tokenize import sent_tokenize
import language_check
from collections import Counter
import pandas as pd
import random
import math
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score

#DIVIDE INTO TRAINING / TESTING

#read in file and generate a random training set to test functions
whole_set = pd.read_csv('train.tsv', delimiter='\t', header= 0) # 17207
train_set= whole_set.sample(n = 16207)
whole_minus_train = whole_set.loc[~whole_set.index.isin(train_set.index)]
validation_set = whole_minus_train.sample(n = 500)
test_set = whole_minus_train.loc[~whole_minus_train.index.isin(validation_set.index)]


#Nino's function that converts text to wordlist
def word_list(text, contraction = False):
    punc_split = re.compile('[' + punctuation.replace("'", "") + ']')
    words = re.sub(punc_split, ' ', text).lower()
    if contraction:
        words = re.sub("[']", " '", words)
    words = words.split()
    return words

#The two functions below are used to create Counter dictionary for each score group
def create_Counter_dict(col_of_strings):
    #big_string = col_of_strings.cat(sep=' ')
    big_string = " ".join(col_of_strings.tolist())
    ct = Counter(word_list(big_string))
    return ct

def get_essays(df, score):
    target_df = train_df[train_df["Score1"] == score]
    essay_col = target_df['EssayText']
    return essay_col

#The two functions below are used when calculating word weight for an essay itself
def frequency(text, keyword):
    wordlist = word_list(text)
    count = wordlist.count(keyword)
    return count

def max_frequency(text):
    wordlist = word_list(text)
    ct = Counter(wordlist)
    max_freq = ct.most_common(1)[0][1]
    return max_freq

#weight for word i in score category s
def word_weight_in_score_category(train_df, score, keyword):
    if keyword in Score_dicts[score]:
        word_freq = Score_dicts[score].get(keyword)
    else:
        word_freq = 0
    maxi_freq = Score_dicts[score].most_common(1)[0][1]
    num_train = len(train_df)
    num_have_word = sum([1 if keyword in t else 0 for t in All_essay])
    #print word_freq, maxi_freq, num_train, num_have_word
    if num_have_word !=0:
        weight = (1.0*word_freq/maxi_freq)*(math.log(1.0*num_train/num_have_word))
    else:
        weight = 0
    return weight

def word_weight_self(train_df, text, keyword):
    word_freq = frequency(text, keyword)
    maxi_freq = max_frequency(text)
    num_train = len(train_df)
    num_have_word = sum([1 if keyword in t else 0 for t in All_essay])
    if num_have_word != 0:
        weight = (1.0 * word_freq / maxi_freq) * (math.log(1.0 * num_train / num_have_word))
    else:
        weight = 0
    return weight

#Main function
def score_point_value(essay, train_df):
    #scorelist = [0,1,2,3] #changed to 'global' scorelist
    wordset = set(word_list(essay))
    word_weight_vectors = [[word_weight_self(train_df,essay,w) for w in wordset]]
    for s in scorelist:
        vector = [word_weight_in_score_category(train_df, s, w) for w in wordset]
        word_weight_vectors.append(vector)
    cosine_corr_matrix = cosine_similarity(word_weight_vectors)
    similarities = cosine_corr_matrix[0]
    similarities = similarities.tolist()
    #The below statement handles essay with only 1 word. Those essays give wrong matrix.
    if similarities[0]== similarities[1]:
        predicted_score = 0
    else:
        predicted_score = similarities.index(max(similarities[1:]))-1
    return predicted_score


for set_id in range(1,11,1):
    train_df = train_set[train_set['EssaySet']==set_id]
    test_df = test_set[test_set['EssaySet']==set_id]
    scorelist = sorted(train_df['Score1'].unique()) #scorelist is different for each essay set

    Score_dicts = [create_Counter_dict(get_essays(train_df, i)) for i in scorelist]
    All_essay0 = train_df['EssayText'].tolist()
    All_essay = [word_list(e) for e in All_essay0]

    predict = [score_point_value(i, train_df) for i in test_df['EssayText'].tolist()]
    print accuracy_score(test_df['Score1'], predict)