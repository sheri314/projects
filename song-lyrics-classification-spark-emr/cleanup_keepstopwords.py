import pandas as pd
import re
import string

data = pd.read_csv('songdata.csv')
data = data[['artist', 'song', 'text']]

def cleantext(line):
    """ Remove new lines,
    sub multiple spaces,
    lower all text,
    remove all punctuation,
    remove stopwords and words less than 3"""
    line = line.strip()
    line = line.replace('\n','')
    line = re.sub(' +', ' ', line)
    line = line.lower()
    line = line.translate(None, string.punctuation)
    line_split = line.split(' ')
    #line_wo_stop = [w for w in line_split if (len(w) > 2) and (w not in stopwords)]
    return ' '.join(line_split)

columns = data.columns

for name in columns:
    data[name] = data[name].map(lambda line : cleantext(line))

data.to_csv('cleanedsongdata.csv')
