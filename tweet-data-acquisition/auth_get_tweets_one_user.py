import sys
import tweepy
import datetime


# ---------------------------------------------------------------------#
#                    FUNCTIONS                                        #
# ---------------------------------------------------------------------#

def tw_auth(consumer_key, consumer_secret, access_token, access_token_secret):
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)

    return tweepy.API(auth)


def get_tweet_cursor(api):
    tweet_text = []
    tweet_hashtags = []
    tweet_dates = []

    for tweet in tweepy.Cursor(api.user_timeline, screen_name='gavinwilliams38', count=5, since_id=2016 - 02 - 07,
                               until=2016 - 02 - 15).items():

        # Criteria for the date range of the tweets. For SuperBowl, from Jan 27, 2016 to Feb 21, 2016
        if tweet.created_at > datetime.datetime(2016, 01, 27, 0, 0, 0) \
                and tweet.created_at < datetime.datetime(2016, 02, 21, 0, 0, 0):
            tweet_text += [tweet.text.encode('utf8')]
            tweet_dates += [str(
                tweet.created_at)]  # tweet.created_at is a datetime.datetime format. Str() converts to string so its easy to read.

            # If more than one hashtag is present in the tweet, this loops through them all
            for i in range(len(tweet._json['entities']['hashtags'])):
                tweet_hashtags += [tweet._json['entities']['hashtags'][i]['text']]

    return tweet_text, tweet_hashtags, tweet_dates


def oops(api):
    data = api.rate_limit_status()

    remain_timeline = data['resources']['statuses']['/statuses/user_timeline']['remaining']
    remain_user = data['resources']['users']['/users/lookup']['remaining']

    print "USER_TIMELINE - REMAINING:", remain_timeline
    print "USERS/LOOKUP - REMAINING:", remain_user

    if remain_user < 25:
        print "Go Take A Break! Only", remain_user, "User Queries Left"
    elif remain_timeline < 25:
        print "Go Take A Break! Only", remain_timeline, "Timeline Queries Left"


# ---------------------------------------------------------------------#
#               CALL FUNCTIONS                                        #
# ---------------------------------------------------------------------#
api = tw_auth(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
the_text, the_hashtags, the_dates = get_tweet_cursor(api)
oops(api)

print "The Text of the Tweets \n", the_text, "\n"
print "The Hashtags of the Tweets \n", the_hashtags, "\n"
print "The Dates for Selected Tweets \n", the_dates, "\n"
