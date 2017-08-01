import tweepy
import sys
from geotext import GeoText
from searchtwitter_fun import getSearch

def cleanLocation(data):
    locationList = []
    for i in range(len(data)):
        for j in range(len(data[i])):
            location = GeoText(data[i][j][7]).cities
            data[i][j].append(location)
            locationList += location
    return data, locationList

if __name__=="__main__":
    outputData = getSearch(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    cleanData, locations = cleanLocation(outputData)
    print locations