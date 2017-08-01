from filter import *

# denoise v.2.0

def median(data):
    data.sort()
    medianindex = len(data)/2
    return data[medianindex]

img = open(sys.argv)
img.show()
img = filter(img, median)
img.show()