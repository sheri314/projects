def getText(results):
    tweet_text = []  # tweets text
    for tweet in results:
        tweet_text.append(tweet.text.encode('utf8'))
    return tweet_text

def getHashtags(results):
    hash_list = []  # list of hashtags
    for tweet in results:
        for tag in tweet.entities['hashtags']:
            hash_list.append(tag['text'].encode('utf8'))
    return hash_list


def getTimes(results):
    tweet_times = []
    for tweet in results:
        tweet_times.append(tweet.created_at)

    return tweet_times

def getMentions(results):
    mentions = []  # returns all users mentioned in all tweets
    for tweet in results:
        for user in tweet.entities['user_mentions']:
            mentions.append(user['screen_name'])

    return mentions

def getLocation(results):
    user_location = []
    for tweet in results:
        user_location.append(tweet.user.location)
    return user_location

def getLikes(results):
    post_likes = []  # returns list of tuples USER ID, FAVORITES COUNT, TEXT
    for tweet in results:
        post_likes.append((tweet.user.id, tweet.favorite_count))

    return post_likes

def getFollowers(results):
    user_followers = []  # returns USER ID, NUMBER OF FOLLOWERS
    for tweet in results:
        user_followers.append((tweet.user.id, tweet.user.followers_count))
    return user_followers

