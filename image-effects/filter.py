import sys
from PIL import Image

# filter.py will be our library file

def open(argv): # opens file
    if len(sys.argv)<=1:
        print "missing image filename"
        sys.exit(1)
    filename = sys.argv[1]
    img = Image.open(filename)
    img = img.convert("L")
    return img

def region3x3(img, x, y): # gets pixel value in 3x3 regions and returns them in a list
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

def getpixel(img, x, y): # retrieves pixel value of image at point (x,y)
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

def filter(img, f): # choose an effect and an image to carry out the effect on
    width, height = img.size
    copyimg = img.copy()
    pixels = copyimg.load()
    for i in range(0, width):
        for j in range(0, height):
            r = region3x3(img, i, j)
            pixels[i,j] = f(r)
    return copyimg



