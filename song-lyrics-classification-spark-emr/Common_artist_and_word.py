#MOST COMMON WORDS BY ARTISTS

from collections import Counter
#Counter(words).most_common(10)
rdd = sc.textFile("cleanedsongdata.csv")


wordlist = rdd.map(lambda x: (x.split(",")[1], (x.split(",")[3]).split(" ")))
word_byartist = wordlist.groupByKey()

#print result
#for i in word_byartist.collect()[1:10]:
#    print i[0]
#    for j in i[1]:
#        print j


word_byartist_merge = word_byartist.mapValues(lambda l: [item for sublist in l for item in sublist])

#print result
#word_byartist_merge.collect()[1:10]

common_20_word = word_byartist_merge.mapValues(lambda l: Counter(l).most_common(20))

#print 10 artists
common_20_word.collect()[1:10]


#MOST COMMON ARTISTS

artist = rdd.map(lambda x: (x.split(",")[1], 1))
artist_count = artist.reduceByKey(lambda x,y: x+y)
most_common_artist = artist_count.sortBy(lambda x: x[1], ascending = False).take(20)
most_common_artist



