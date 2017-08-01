from string import punctuation
import re
from nltk.tokenize import sent_tokenize
import language_check
from collections import Counter
import pandas as pd
import random
import math
from sklearn.metrics.pairwise import cosine_similarity

############# From Nino's ###############

#read in file and generate a random training set to test functions
text_df = pd.read_csv('train.tsv', delimiter= '\t', header= 0)
random.seed(4)
train_X_indices = random.sample(range(len(text_df)), 300)
train_X = text_df.loc[train_X_indices,:]
#print train_X
#print train_X['EssayText']

def word_list(text, contraction = False):
    punc_split = re.compile('[' + punctuation.replace("'", "") + ']')
    words = re.sub(punc_split, ' ', text).lower()
    if contraction:
        words = re.sub("[']", " '", words)
    words = words.split()
    return words

#############################################

#This is an essay I took from the random sample, to test functions
test_str = "The cellular respiration process controls the water flow throughout a cell membrane, the mRNA transports messages throughout the cell, and the tRNA produces proteins."

############# A FEW SMALL FUNCTIONS ############

#Average Word Length
def avg_word_length(text):
    words = word_list(text)
    word_length = [len(w) for w in words]
    avg_word_length = sum(word_length)*1.0/len(words)
    return avg_word_length

#Sentence length
def sentence_length(text):
    sent_tokenize_list = sent_tokenize(text)
    sent_length = len(sent_tokenize_list)
    return sent_length

#Language mistake count (including spelling mistakes and grammar mistakes)
def language_mistake_count(text):
    tool = language_check.LanguageTool('en-US')
    matches = tool.check(text)
    return len(matches)

#Given a wordlist, return 'top' number of popularwords
def popularWords(text, top):
    wordlist = word_list(text)
    count = Counter(wordlist)
    popular_words = sorted(count, key=count.get, reverse=True)
    top_words = popular_words[:top]
    return top_words

###########################################



####### Score point value (1 feature) ###############
#details in https://www.ets.org/Media/Research/pdf/RR-04-45.pdf

#Frequency: given a word, calculate how often the word apprears in a list of wordlists
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
def word_weight_in_score_category(train_df, score_category, keyword):
    #df could be training data frame or testing data frame
    target_df = train_df[train_df["Score1"]==score_category]
    #print target_df
    target_df = target_df['EssayText']
    text_list = target_df.tolist() #convert pandas series to a list of texts
    merged_text = " ".join(text_list) #merge texts to a long string
    word_freq = frequency(merged_text,keyword)
    num_train = len(train_df)
    maxi_freq = max_frequency(merged_text)
    all_essay = train_df['EssayText'].tolist()
    num_have_word = sum([1 if keyword in t else 0 for t in all_essay])
    if num_have_word !=0:
        weight = (1.0*word_freq/maxi_freq)*(math.log(1.0*num_train/num_have_word))
    else:
        weight = 0
    return weight

def word_weight_self(train_df, text,keyword):
    word_freq = frequency(text, keyword)
    maxi_freq = max_frequency(text)
    num_train = len(train_df)
    all_essay = train_df['EssayText'].tolist()
    num_have_word = sum([1 if keyword in t else 0 for t in all_essay])
    if num_have_word != 0:
        weight = (1.0 * word_freq / maxi_freq) * (math.log(1.0 * num_train / num_have_word))
    else:
        weight = 0
    return weight

#Main function
def score_point_value(essay, train_df):
    # df could be training data frame or testing data frame
    scorelist = [0,1,2,3]
    wordset = set(word_list(essay))
    word_weight_vectors = [[word_weight_self(train_df,essay,w) for w in wordset]]
    for s in scorelist:
        vector = [word_weight_in_score_category(train_df, s, w) for w in wordset]
        word_weight_vectors.append(vector)
    cosine_corr_matrix = cosine_similarity(word_weight_vectors)
    similarities = cosine_corr_matrix[0]
    similarities = similarities.tolist()
    return similarities.index(max(similarities[1:]))

print score_point_value(test_str, train_X)

#####################