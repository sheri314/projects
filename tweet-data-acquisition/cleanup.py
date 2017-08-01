import tweepy
import sys
import csv
from searchtwitter_fun import getSearch
from locations import cleanLocation

def mush(data):
    csvList = ['Text,UserID,Hashtags,Mentions,Likes,Date,FollowerCount,Location,CleanLocation']
    data = cleanLocation(data)[0]
    csvData = data
    for i in range(len(data)):
        for j in range(len(data[i])):
            csvData[i][j][1] = str(data[i][j][1])
            try:
                csvData[i][j][2] = ';'.join(data[i][j][2])
            except:
                pass
            try:
                csvData[i][j][3] = ';'.join(data[i][j][3])
            except:
                pass
            csvData[i][j][4] = str(data[i][j][4])
            csvData[i][j][5] = str(data[i][j][5])
            tweetLine = ','.join(csvData[i][j])
            csvList.append(tweetLine)
        mycsv = '\n'.join(csvList)
    return mycsv

def writecsv(data, loc, fname):
    name = loc + '/' + fname + '.csv'
    header = ['Text','UserID','Hashtags','Mentions', 'Likes','Date','FollowerCount','Location', 'CleanLocation']
    data = cleanLocation(data)[0]
    csvData = data
    try:
        file = open(name, 'wb')
    except:
        print 'oops!'
    pencil = csv.writer(file)
    pencil.writerow(header)
    for i in range(len(data)):
        for j in range(len(data[i])):
            try:
                csvData[i][j][2] = ';'.join(data[i][j][2])
            except:
                pass
            try:
                csvData[i][j][3] = ';'.join(data[i][j][3])
            except:
                pass
            for k in range(len(csvData[i][j])):
                try:
                    csvData[i][j][k] = csvData[i][j][k].encode('utf8')
                except:
                    pass
            pencil.writerow(csvData[i][j])
    file.close()

if __name__ == '__main__':
    outputData = getSearch(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    outputDir = sys.argv[6]  # where to save output file (e.g. 'Output' will save to a file in same directory called 'Output')
    filename = sys.argv[7]  # what to name the file (if there is already a file in this location with this name, it will be overwritten)
    writecsv(outputData, outputDir, filename)