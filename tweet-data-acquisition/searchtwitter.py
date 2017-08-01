import tweepy
import sys
import time
from geotext import GeoText
from parsetweet import *
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from text import *
import csv

def makeDataRequest(term, retMinID):
    results = myapi.search(q = term, count = 100, max_id = retMinID, lang = 'en')
    return results

def newMaxID(result):
    topID = result[-1].id
    return topID

def getSearch(myapi, searchTerm, numIter):
    all_text = []
    all_hashtags = []
    all_times = []
    all_mentions = []
    all_locations = []
    all_likes = []
    all_followers = []

    topID = ''

    for i in range(numIter):
        if i == 1:
            Search = myapi.search(q=searchTerm, count=100, lang='en')
            topID = newMaxID(Search)
        else:
            try:
                Search = makeDataRequest(searchTerm, topID)
                topID = newMaxID(Search)
            except tweepy.TweepError:
                time.sleep(60*15)
                print ("You've reached the rate limit. Program will resume in 15 minutes.")
                continue
        all_text += getText(Search)
        all_hashtags += getHashtags(Search)
        all_times += getTimes(Search)
        all_mentions += getMentions(Search)
        all_locations += getLocation(Search)
        all_likes += getLikes(Search)
        all_followers += getFollowers(Search)

    allreturns = [nostopwords(tokenize(all_text)), nostopwords(tokenize(all_hashtags)), all_times,
                  nostopwords(tokenize(all_mentions)), all_locations, all_likes, all_followers]

    return allreturns

def cleanLocation(data):
    locationList = []
    for spot in data:
        location = GeoText(spot).cities # extract city from strings
        if len(location) != 0:
            locationList += location
    return locationList

def makeCloud(wordDict):
    wordcloud = WordCloud(width=1800,
                          height=1400,
                          max_words=10000,
                          random_state=1,
                          relative_scaling=0.25)

    wordcloud.fit_words(wordDict)

    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()


def mainFunc(myapi, term, numIter, topN):
    results = getSearch(myapi, term, numIter)
    text = results[0]
    hashtags = results[1]
    times = results[2] # FOR FURTHER ANALYSIS
    mentions = results[3]
    locations = cleanLocation(results[4])
    likes = results[5]
    followers = results[6]

    mostText = top(text, topN)
    mostHashtags = top(hashtags, topN)
    mostMentions = top(mentions, topN)
    mostLocations = top(locations, topN)
    mostLikes = orderTup(likes, topN)
    mostFollowers = orderTup(followers, topN)

    return mostText, mostHashtags, mostMentions, mostLocations, mostLikes, mostFollowers

if __name__ == '__main__':
    consumer_key = sys.argv[1]
    consumer_secret = sys.argv[2]
    access_token = sys.argv[3]
    access_token_secret = sys.argv[4]

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    myapi = tweepy.API(auth)

    term = sys.argv[5]
    numIter = int(sys.argv[6]) # this number times 100 is the number of tweets returned
    if numIter >= 450:
        sys.exit("This exceeds the rate limit, please rerun for number of iterations below 450.")
    topN = int(sys.argv[7]) # the number in top returns
    text, hashtags, mentions, locations, likes, followers = mainFunc(myapi, term, numIter, topN)
    allCounts = [text, hashtags, mentions, locations, likes, followers]
    filenames = ['text', 'hashtags', 'mentions', 'locations', 'likes', 'followers']
    # CAN MAKE A CLOUD
    # for object in allCounts:
    #     makeCloud(object)
    # OR WRITE THE ELEMENTS TO A DOCUMENT
    for fname in range(len(filenames)):
        with open(filenames[fname] + '.csv', 'wb') as f:
            writer = csv.writer(f)
            for row in range(len(allCounts[fname])):
                writer.writerow((allCounts[fname][row][0], allCounts[fname][row][1]))

    print 'done'




