import tweepy
import string
import re
import sys
# from nltk import stem
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction import stop_words
from collections import Counter

# KEYS AND TOKENS NEEDED FOR AUTHORIZATION
consumer_key= sys.argv[2]
consumer_secret= sys.argv[3]

access_token= sys.argv[4]
access_token_secret= sys.argv[5]

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)


status = api.rate_limit_status() # check rate limit statuses

def getInfo(results):
    tweets = []
    tweet_text = []
    hash_list = []

    for tweet in results:

        tweet_text.append(tweet.text.encode('utf8'))

        for tag in tweet.entities['hashtags']:
            hash_list.append(tag['text'].encode('utf8'))

        # hashtags = ",".join(h for h in hash_list)

        mentions = []
        for user in tweet.entities['user_mentions']:
            mentions.append(user['screen_name'])

        #All Tweet Info
        tweets.append([tweet.text,
                       tweet.user.id,
                       # hashtags,
                       mentions,
                       tweet.created_at,
                       tweet.user.followers_count,
                       tweet.user.location])
    return tweet_text, hash_list
    #return tweets

def makeDataRequest(term, retMinID):
    results = api.search(q = term, count = 100, max_id = retMinID)
    return results

def newMaxID(result):
    topID = result[-1].id
    return topID

def tokenize(text):
    text2 = " ".join(text)
    bad_chars = re.compile('[%s + 0-9\\r\\t\\n]' % re.escape(string.punctuation))
    clean_text = re.sub(bad_chars, ' ', text2.lower())

    words = word_tokenize(clean_text)

    long_words = []

    for w in words:
        if len(w) >= 3:
            long_words.append(w)

    return long_words

def nostopwords(words):
    no_stop_words = [w for w in words if w not in stop_words.ENGLISH_STOP_WORDS]

    return no_stop_words

def tweets_with_search_term(searchTerm, numIter):
     # returns 300 tweets and the necessary info
    output_text = []
    output_hashtags = []
    topID = ''

    for i in range(numIter): # minus 1 to account for the initial search
        if i == 1:
            Search = api.search(q=searchTerm, count=100, lang='en')
            topID = newMaxID(Search)
        else:
            Search = makeDataRequest(searchTerm, topID)
            topID = newMaxID(Search)

        # RETURN LIST OF TWEET TEXT AND HASHTAG TEXT
        tweet_text, hashtags = getInfo(Search)

        # CONVERT LIST TO TEXT, REMOVE BAD CHARS, MAKE LOWERCASE AND DELETE WORDS <3 LETTERS
        # REMVOVE ENGLISH STOPWORDS
        token_text = tokenize(tweet_text)
        output_text += nostopwords(token_text)

        token_hashtags = tokenize(hashtags)
        output_hashtags += nostopwords(token_hashtags)

    # CREATE COUNTER FROM FINAL CLEANED WORDS
    counts_text = Counter(output_text)
    counts_hashtags = Counter(output_hashtags)

    # OUTPUT CURRENTLY PRINTING. USE RETURN STATEMENT IF WE NEED TO USE FOR SOMETHING ELSE
    print "OUTPUT TEXT \n", output_text, "\n"
    print "OUTPUT HASHTAGS \n", output_hashtags, "\n"

    print "TEXT COUNTS \n", counts_text, "\n"
    print "HASHTAG COUNTS \n", counts_hashtags, "\n"

if __name__=="__main__":
    # searchTerm = 'fleetweeksf'
    tweets_with_search_term(searchTerm=sys.argv[1], numIter= 3)