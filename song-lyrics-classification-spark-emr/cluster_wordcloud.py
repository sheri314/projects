from wordcloud import WordCloud
import matplotlib.pyplot as plt
import sys
from collections import Counter
import pandas as pd

#the target cluster number is a commandline input
cluster_number = int(sys.argv[1])
file = 'clusterDF.csv'
#cluster_number = 8

#read in the stopwords
text_file = open("stopwords", "r")
lines = text_file.read().splitlines()
stopwords = []
for i in lines:
    stopwords.append(i)

#If want to remove "yo" uncomment this
#stopwords.append("yo")

df = pd.read_csv(file)
target_cluster = df.loc[df['cluster']==cluster_number]
cluster_word = target_cluster['text'].tolist()

#made word list
words = []
for i in range(0, len(cluster_word)):
    words = words + (cluster_word[i].replace("[", "").replace("]", "").replace("u","").replace("'","")).split(", ")

#remove stopwords
words_nostopwords = [w for w in words if w not in stopwords]
data = Counter(words_nostopwords)

#test
#data = Counter(["no","random","sql","time","hi","hello"])

wordcloud = WordCloud(width=1800, height=1400, max_words=10000, random_state=1, relative_scaling=0.25)

wordtuples = []
for (k, v) in data.iteritems():
    tuple = (k,v)
    wordtuples.append(tuple)

print sorted(wordtuples, key=lambda x: x[1], reverse=True)

wordcloud.fit_words(wordtuples)

plt.imshow(wordcloud)
plt.axis("off")
plt.show()