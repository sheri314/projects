import tweepy
import sys

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
                       tweet.favorite_count,
                       tweet.created_at,
                       tweet.user.followers_count,
                       tweet.user.location])
    return tweets

def makeDataRequest(myapi, term, retMinID):
    results = myapi.search(q = term, count = 100, max_id = retMinID)
    return results

def newMaxID(result):
    topID = result[-1].id
    return topID

def getSearch(consumer_key, consumer_secret, access_token, access_token_secret, searchTerm):
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)

    api = tweepy.API(auth)

    status = api.rate_limit_status()  # check rate limit statuses
    initialSearch = api.search(q = searchTerm, count = 100, lang = 'en')
    topID = newMaxID(initialSearch)
    numIter = 3 # returns 300 tweets and the necessary info
    outputData = []
    outputData.append(getInfo(initialSearch))
    for i in range(numIter - 1): # minus 1 to account for the initial search
        searchSet = makeDataRequest(api, searchTerm, topID)
        outputData.append(getInfo(searchSet))
        topID = newMaxID(searchSet)
    #print outputData #return a list of lists of searches
    return outputData

if __name__ == '__main__':
    output = getSearch(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4], sys.argv[5])
    print output