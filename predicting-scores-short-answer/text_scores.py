
# coding: utf-8

# In[115]:

from string import punctuation
import pandas as pd
import numpy as np
import re
from collections import Counter
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
import random
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn import tree
from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# In[101]:

text_df = pd.read_csv('train.tsv', delimiter= '\t', header= 0)
EssaySet_dummies = pd.get_dummies(data= text_df.loc[:, ['Id', 'EssaySet']], columns= ['EssaySet'], prefix= 'Set')
text_df = text_df.merge(EssaySet_dummies, on= ['Id'], how= 'left')
print list(text_df.columns.values)


# In[102]:

def word_list(text, contraction = False):
    punc_split = re.compile('[' + punctuation.replace("'", "") + ']')
    words = re.sub(punc_split, ' ', text).lower()
    if contraction:
        words = re.sub("[']", " '", words)
    words = words.split()
    return words

def punc_list(text):
    punc = re.findall('[.,?!;:]', text)
    return punc

def word_rate(w, text):
    words = word_list(text)
    count = words.count(w)
    return count/(len(words)*1.0)

def spell_rate(text):
    length = len(word_list(text, True) + punc_list(text))
    words = word_list(text)
    count = 0
    for word in words:
        if word not in dictionary_list:
            count += 1
    return count/(length * 1.0)

def word_count(series):
    words = []
    for i in range(len(series)):
        words.extend([w for w in word_list(series.iloc[i]) if w not in ENGLISH_STOP_WORDS])
    return Counter(words)

def distinct_rate(text):
    words = word_list(text)
    return len(set(words))/(len(words)*1.0)


# In[103]:

Common_words = {}
text_set = text_df[['EssaySet', 'Score1', 'EssayText']]
for i in set(text_set['EssaySet']):
    score_text = {}
    text_subset = text_set[(text_set.EssaySet == i)]
    for j in set(text_subset['Score1']):
        score_text[j] = word_count(text_subset[(text_subset.Score1 == j)]['EssayText'])
    Common_words[i] = score_text

new_words = []
for i in range(1,11):
    common = Common_words[i][max(text_set[(text_set.EssaySet == i)]['Score1'])].most_common(10)
    new_words.extend([w for w,v in common if len(w)>2 and w not in punctuation])    
    
len(set(new_words))


# In[104]:

dictionary_list = set()
with open('/Users/lawrencebarrett/lgbarrett/ML_Project/Test_scores/english_dic/eng_com.dic', 'r') as eng_dictionary:
    lines = eng_dictionary.readlines()
    for line in lines:
        dictionary_list.add(line.strip())


# In[105]:

for word in set(new_words):
    text_df[word] = text_df['EssayText'].apply(lambda x: word_rate(word, x))

text_df['Spelling'] = text_df['EssayText'].apply(lambda x: spell_rate(x))
text_df['Distinct_words'] = text_df['EssayText'].apply(lambda x: distinct_rate(x))
text_df['Text_Length'] = text_df['EssayText'].apply(lambda x: len(word_list(x)))
text_df['Sentences'] = text_df['EssayText'].apply(lambda x: len(re.findall('[.?!]', x)))


# In[106]:

print text_df.head(10)


# In[52]:

len(text_df)


# In[118]:

# Decision Tree
random.seed(4)
test_indices = random.sample(range(len(text_df)), 3000)
train_indices = [i for i in range(len(text_df)) if i not in test_indices]

train_X = text_df.loc[train_indices, 'Set_1':'Sentences']
train_Y = text_df.loc[train_indices, ['Score1']]

test_X = text_df.loc[test_indices, 'Set_1':'Sentences']
test_Y = text_df.loc[test_indices, ['Score1']]

# Cross Validation
crit = 'gini'
classifier = tree.DecisionTreeClassifier(criterion=crit, random_state=0, max_depth=5)
kfold = KFold(n_splits=20, shuffle=True, random_state=0)
scores = cross_val_score(classifier, train_X, train_Y, cv=kfold)
print 1 - np.mean(scores)

# Test
tree_classifier = classifier.fit(train_X, train_Y)
score = tree_classifier.score(test_X, test_Y)
print 1 - score


# In[114]:

# Linear SVC
random.seed(4)
test_indices = random.sample(range(len(text_df)), 3000)
train_indices = [i for i in range(len(text_df)) if i not in test_indices]

train_X = text_df.loc[train_indices, 'Set_1':'Sentences']
train_Y = text_df.loc[train_indices, ['Score1']]

test_X = text_df.loc[test_indices, 'Set_1':'Sentences']
test_Y = text_df.loc[test_indices, ['Score1']]

# Cross Validation
classifier = LinearSVC()
kfold = KFold(n_splits=20, shuffle=True, random_state=0)
scores = cross_val_score(classifier, train_X, np.array(train_Y), cv=kfold)
print 1 - np.mean(scores)

# Test
svc_classifier = classifier.fit(train_X, train_Y)
score = svc_classifier.score(test_X, test_Y)
print 1 - score


# In[119]:

# LDA
random.seed(4)
test_indices = random.sample(range(len(text_df)), 3000)
train_indices = [i for i in range(len(text_df)) if i not in test_indices]

train_X = text_df.loc[train_indices, 'Set_2':'Sentences']
train_Y = text_df.loc[train_indices, ['Score1']]

test_X = text_df.loc[test_indices, 'Set_2':'Sentences']
test_Y = text_df.loc[test_indices, ['Score1']]

# Cross Validation
classifier = LinearDiscriminantAnalysis()
kfold = KFold(n_splits=20, shuffle=True, random_state=0)
scores = cross_val_score(classifier, train_X, np.array(train_Y), cv=kfold)
print 1 - np.mean(scores)

# Test
lda_classifier = classifier.fit(train_X, train_Y)
score = lda_classifier.score(test_X, test_Y)
print 1 - score


# In[ ]:



