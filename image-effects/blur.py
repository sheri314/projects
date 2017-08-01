import sys
from PIL import Image

# blur file v.1.0

def blur(filename): # blurs loaded image
    width, height = img.size
    copyimg = img.copy() # creates a copy of original image
    pixels = copyimg.load() # creates a 2D matrix of pixel values of copied image line #7
    for i in range(0, width):
        for j in range(0, height):
            r = region3x3(img, i, j)
            pixels[i,j] = avg(r)
    return copyimg #returns blurred image

def region3x3(img,x,y): # gets pixel value in 3x3 regions and returns them in a list
    me = getpixel(img,x,y)
    N = getpixel(img,x,y - 1)
    S = getpixel(img, x, y + 1)
    E = getpixel(img, x + 1, y)
    W = getpixel(img, x - 1, y)
    NW = getpixel(img, x - 1, y - 1)
    NE = getpixel(img, x + 1, y - 1)
    SE = getpixel(img, x + 1, y + 1)
    SW = getpixel(img, x - 1, y + 1)
    return [me, N, S, E, W, NW, NE, SE, SW]

def avg(data): # returns the average of the data returned in function region3x3
    return sum(data)/len(data)

def getpixel(img,x,y): # retrieves pixel value of image at point (x,y)
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


if len(sys.argv)<=1: # if no file is pass through, message will appear
    print "missing image filename"
    sys.exit(1)
filename = sys.argv[1] # accepts file as argument
img = Image.open(filename) # opens file
img = img.convert("L") # converts image to 8-bit pixels (black and white)
img.show() # shows original image
blur(filename).show() # shows image after blue filter
