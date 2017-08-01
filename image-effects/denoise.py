import sys
from PIL import Image

# denoise file v.1.0

def denoise(filename): # denoise a noisy image
    width, height = img.size
    copyimg = img.copy()
    pixels = copyimg.load()
    for i in range(0, width):
        for j in range(0, height):
            r = region3x3(img, i, j)
            pixels[i, j] = median(r)
    return copyimg

def region3x3(img, x, y):
    me = getpixel(img, x, y)
    N = getpixel(img, x, y - 1)
    S = getpixel(img, x, y + 1)
    E = getpixel(img, x + 1, y)
    W = getpixel(img, x - 1, y)
    NW = getpixel(img, x - 1, y - 1)
    NE = getpixel(img, x + 1, y - 1)
    SE = getpixel(img, x + 1, y + 1)
    SW = getpixel(img, x - 1, y + 1)
    return [me, N, S, E, W, NW, NE, SE, SW]

def median(data): # takes list returned by region3x3, sorts list, finds median of list, and returns value at that index
    data.sort()
    medianindex = len(data)/2
    return data[medianindex]


def getpixel(img, x, y):
    width, height = img.size
    originalpixels = img.load()
    if x < 0:
        x = 0
    if x >= width:
        x = width - 1
    if y < 0:
        y = 0
    if y >= height:
        y = height - 1
    return originalpixels[x,y]


if len(sys.argv)<=1:
	print "missing image filename"
	sys.exit(1)
filename = sys.argv[1]
img = Image.open(filename)
img = img.convert("L")
img.show()
denoise(filename).show() # shows image after passing through denoise function

