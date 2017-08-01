import string
import re
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction import stop_words
from collections import Counter

def tokenize(text):
    text2 = " ".join(text)
    bad_chars = re.compile('[%s + 0-9\\r\\t\\n]' % re.escape(string.punctuation))
    clean_text = re.sub(bad_chars, ' ', text2.lower())
    words = word_tokenize(clean_text)
    long_words = []
    for w in words:
        if len(w) >= 3 and w != 'https':
            long_words.append(w)
    return long_words

def nostopwords(words):
    no_stop_words = [w for w in words if w not in stop_words.ENGLISH_STOP_WORDS]
    return no_stop_words

def top(wordList, number):
    """
    RETURN COUNTER OF TOP number WORD INSTANCES
    """
    histo = Counter()
    for word in (wordList):
        histo[word] += 1
    return histo.most_common(number)

def orderTup(wordList, number):
    return sorted(wordList, key = lambda x : x[1], reverse = True)[:number]





