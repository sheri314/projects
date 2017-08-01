from filter import *

#blur v.2.0

def avg(data): # returns the average of the data returned in function region3x3
    return sum(data)/len(data)

img = open(sys.argv)
img.show()
img = filter(img, avg)
img.show()
