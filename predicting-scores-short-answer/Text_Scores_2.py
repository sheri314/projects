from string import punctuation
import pandas as pd
import numpy as np
import re
from collections import Counter
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
import random
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
# import language_check
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize
import warnings

warnings.filterwarnings("ignore")

def word_list(text, contraction = True):
    # Returns words from text. If contraction = False, words like "don't" are split
    # into "don" and "'t"
    punc_split = re.compile('[' + punctuation.replace("'", "") + ']')
    words = re.sub(punc_split, ' ', text).lower()
    if not contraction:
        words = re.sub("[']", " '", words)
    words = words.split()
    return words

def punc_list(text):
    # Returns punctuation from text
    punc = re.findall('[.,?!;:]', text)
    return punc

def word_rate(w, text):
    # Computes rate of appearance of word w in text
    words = word_list(text)
    count = words.count(w)
    return count/(len(words)*1.0)

def spelling_mistakes(text):
    # Computes number of misspellings
    words = word_list(text)
    count = 0
    for word in words:
        if word not in dictionary_list:
            count += 1
    return count

def word_count(series):
    # Computes count of unique words in a column
    words = []
    for i in range(len(series)):
        words.extend([w for w in word_list(series.iloc[i]) if w not in ENGLISH_STOP_WORDS])
    return Counter(words)

def distinct_rate(text):
    # Computer rate of distinct words
    words = word_list(text)
    return len(set(words))/(len(words)*1.0)

#---------------------------------------------
# Su's Functions
def avg_word_length(text):
    # Computes average word length
    words = word_list(text)
    word_length = [len(w) for w in words]
    avg_word_length = sum(word_length)/float(len(words))
    return avg_word_length

def sentence_length(text):
    # Counts the number of sentences in text
    sent_tokenize_list = sent_tokenize(text)
    sent_length = len(sent_tokenize_list)
    return sent_length

def language_mistake_count(text):
    # Counts the number of language mistakes including spelling and grammar mistakes
    tool = language_check.LanguageTool('en-US')
    matches = tool.check(text)
    return len(matches)

def popularWords(text, n):
    # Returns list of top n popular words in a text
    wordlist = word_list(text)
    count = Counter(wordlist)
    popular_words = sorted(count, key=count.get, reverse=True)
    top_words = popular_words[:n]
    return top_words

#---------------------------------------------
# Score Similarity Function

#The two functions below are used to create Counter dictionary for each score group

def create_Counter_dict(col_of_strings):
    # Returns dictionary of word counts for a column/series of strings
    #big_string = col_of_strings.cat(sep=' ')
    big_string = " ".join(col_of_strings.tolist())
    ct = Counter(word_list(big_string))
    return ct

def get_essays(train_df, score):
    # Subsets train_df for scores equal to the input score value
    target_df = train_df[train_df["Score1"] == score]
    essay_col = target_df['EssayText']
    return essay_col

#The two functions below are used when calculating word weight for an essay itself

def frequency(text, keyword):
    # Returns the number of times keyword shows up in text
    wordlist = word_list(text)
    count = wordlist.count(keyword)
    return count

def max_frequency(text):
    # Returns the maximum frequency for all words in text
    wordlist = word_list(text)
    ct = Counter(wordlist)
    max_freq = ct.most_common(1)[0][1]
    return max_freq

def word_weight_in_score_category(train_df, score, keyword, Score_dicts, all_essay):
    # Computes the weight of keyword in a specific score category
    if keyword in Score_dicts[score]:
        word_freq = Score_dicts[score].get(keyword)
    else:
        word_freq = 0
    maxi_freq = Score_dicts[score].most_common(1)[0][1]
    num_train = len(train_df)
    num_have_word = np.sum([1 if keyword in t else 0 for t in all_essay])
    #print word_freq, maxi_freq, num_train, num_have_word
    if num_have_word !=0 and maxi_freq !=0:
        weight = (word_freq/float(maxi_freq))*(np.log(num_train/float(num_have_word)))
    else:
        weight = 0
    return weight

def word_weight_self(train_df, text, keyword, all_essay):
    # Computes the weight of keyword in text
    word_freq = frequency(text, keyword)
    maxi_freq = max_frequency(text)
    num_train = len(train_df)
    num_have_word = np.sum([1 if keyword in t else 0 for t in all_essay])
    if num_have_word != 0 and maxi_freq !=0:
        weight = (word_freq / float(maxi_freq)) * (np.log(num_train / float(num_have_word)))
    else:
        weight = 0
    return weight

#Main function
def score_point_value(essay, train_df, scorelist, Score_dicts, all_essay):
    # Assigns a score to an essay based on the weight of each word in that essay and the weights of
    # each word in essays for different scores
    wordset = set(word_list(essay))
    word_weight_vectors = [[word_weight_self(train_df, essay, w, all_essay) for w in wordset]]
    for s in scorelist:
        vector = [word_weight_in_score_category(train_df, s, w, Score_dicts, all_essay) for w in wordset]
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

# Reading in Data
text_df = pd.read_csv('train.tsv', delimiter= '\t', header= 0)

# We had trouble installing Language Check on all our machines, so we had one machine produce a
# csv of the results for use in our features.
mistakes = pd.read_csv('mistake_count.csv', header= 0)

# Creating dictionary list for spell check
dictionary_list = set()
with open('eng_com.dic', 'r') as eng_dictionary:
    lines = eng_dictionary.readlines()
    for line in lines:
        dictionary_list.add(line.strip())

# Feature extraction
text_df['Spelling_Mistakes'] = text_df['EssayText'].apply(lambda x: spelling_mistakes(x))
text_df['Distinct_words'] = text_df['EssayText'].apply(lambda x: distinct_rate(x))
text_df['Text_Length'] = text_df['EssayText'].apply(lambda x: len(word_list(x)))
text_df['Sentence_length'] = text_df['EssayText'].apply(lambda x: sentence_length(x))
text_df['Grammar_Mistakes'] = mistakes['Mistake_Count'].subtract(text_df['Spelling_Mistakes'])
text_df['Avg_word_length'] = text_df['EssayText'].apply(lambda x: avg_word_length(x))

# Building train set and test set
random.seed(4)
test_indices = random.sample(range(len(text_df)), 3000)
train_indices = [i for i in range(len(text_df)) if i not in test_indices]

text_train = text_df.loc[train_indices,:]
text_test = text_df.loc[test_indices, :]

# Score Point Value

# Score_points_train = []
# for j in range(1,11):
#     train_df = text_train[text_train['EssaySet'] == j]
#     scorelist = sorted(train_df['Score1'].unique())
#     Score_dicts = [create_Counter_dict(get_essays(train_df, i)) for i in scorelist]
#     All_essay_wordlists = [set(word_list(e)) for e in train_df['EssayText'].tolist()]
#     score_point_train = [score_point_value(essay, train_df, scorelist, Score_dicts, All_essay_wordlists) for essay in train_df['EssayText'].tolist()]
#     Score_points_train += score_point_train
#     print 'train', j
#
# Score_points_test = []
# for j in range(1,11):
#     test_df = text_test[text_test['EssaySet'] == j]
#     scorelist = sorted(test_df['Score1'].unique())
#     Score_dicts = [create_Counter_dict(get_essays(test_df, i)) for i in scorelist]
#     All_essay_wordlists = [set(word_list(e)) for e in test_df['EssayText'].tolist()]
#     score_point_test = [score_point_value(essay, test_df, scorelist, Score_dicts, All_essay_wordlists) for essay in test_df['EssayText'].tolist()]
#     Score_points_test += score_point_test
#     print 'test', j
#
#
# text_train['Score_point'] = Score_points_train
# text_test['Score_point'] = Score_points_test

# Model fitting and accuracy scoring
classifier_names = ['LDA', 'QDA', 'Linear SVM', 'Decision Tree', 'Random Forest']
classifiers = [LinearDiscriminantAnalysis(), QuadraticDiscriminantAnalysis(), SVC(kernel='linear'),
               DecisionTreeClassifier(criterion='gini', random_state=0, max_depth=5), RandomForestClassifier()]


scores = dict()
for k in range(len(classifiers)):
    print 'fitting %s classifier' % classifier_names[k]
    classifier = classifiers[k]
    scores[classifier_names[k]] = ([], [])
    # score_train_list = []
    # score_test_list = []
    for i in range(1, 11):
        train = text_train[text_train['EssaySet'] == i]
        test = text_test[text_test['EssaySet'] == i]
        #     print train.shape
        #     print test.shape

        train_X = train.loc[:, 'Distinct_words':'Avg_word_length']
        train_Y = train.loc[:, ['Score1']]
        #     print train_X.shape
        #     print train_Y.shape

        test_X = test.loc[:, 'Distinct_words':'Avg_word_length']
        test_Y = test.loc[:, ['Score1']]
        #     print test_X.shape
        #     print test_Y.shape

        classifier.fit(train_X, np.ravel(train_Y))
        # kfold = KFold(n_splits=10, shuffle=True, random_state=0)
        # scores_list = cross_val_score(classifier, train_X, train_Y, cv=kfold)
        # scores[classifier_names[k]][0].append(np.mean(scores_list))

        pred_train = classifier.predict(train_X)
        pred_test = classifier.predict(test_X)

        score_train = accuracy_score(train_Y, pred_train)
        score_test = accuracy_score(test_Y, pred_test)
        scores[classifier_names[k]][0].append(score_train)
        scores[classifier_names[k]][1].append(score_test)
        print '.',
    print 'done'

# print 'Training Accuracy'
# for name in classifier_names:
#     print '\t' + name
#     score_train_list = scores[name][0]
#     for i in range(len(score_train_list)):
#         print '\t\tEssay Set %d:' % (i + 1), score_train_list[i]
#
# print 'Testing Accuracy'
# for name in classifier_names:
#     print '\t' + name
#     score_test_list = scores[name][1]
#     for i in range(len(score_test_list)):
#         print '\t\tEssay Set %d:' % (i + 1), score_test_list[i]


# Printing Results
train_results = [(i+1, scores['LDA'][0][i], scores['QDA'][0][i], scores['Linear SVM'][0][i],
                     scores['Decision Tree'][0][i], scores['Random Forest'][0][i]) for i in range(10)]

test_results = [(i+1, scores['LDA'][1][i], scores['QDA'][1][i], scores['Linear SVM'][1][i],
                     scores['Decision Tree'][1][i], scores['Random Forest'][1][i]) for i in range(10)]

train_results_df = pd.DataFrame(train_results, columns = ['Essay Set'] + classifier_names)
test_results_df = pd.DataFrame(test_results, columns = ['Essay Set'] + classifier_names)

print
print 'Training Accuracy'

print train_results_df

print
print 'Testing Accuracy'

print test_results_df