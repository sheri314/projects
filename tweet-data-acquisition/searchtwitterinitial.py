import tweepy
import sys

consumer_key= ""
consumer_secret= ""

access_token= ""
access_token_secret= ""

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)


status = api.rate_limit_status() # check rate limit statuses

def getInfo(results):
    tweets = []
    for tweet in results:
        hashtags = []
        for tag in tweet.entities['hashtags']:
            hashtags.append(tag['text'])

        mentions = []
        for user in tweet.entities['user_mentions']:
            mentions.append(user['screen_name'])

        tweets.append([tweet.text,
                       tweet.user.id,
                       hashtags,
                       mentions,
                       tweet.created_at,
                       tweet.user.followers_count,
                       tweet.user.location])
    return tweets

def makeDataRequest(term, retMinID):
    results = api.search(q = term, count = 100, max_id = retMinID)
    return results

def newMaxID(result):
    topID = result[-1].id
    return topID

if __name__=="__main__":
    # searchTerm = 'fleetweeksf'
    searchTerm = sys.argv[1]
    initialSearch = api.search(q = searchTerm, count = 100, lang = 'en')
    topID = newMaxID(initialSearch)
    numIter = 3 # returns 300 tweets and the necessary info
    outputData = []
    outputData.append(getInfo(initialSearch))
    for i in range(numIter - 1): # minus 1 to account for the initial search
        searchSet = makeDataRequest(searchTerm, topID)
        outputData.append(getInfo(searchSet))
        topID = newMaxID(searchSet)
    # print outputData #return a list of lists of searches
