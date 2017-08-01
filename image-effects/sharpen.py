from filter import *

def laplace(data):
    laplacevalue = data[1] + data[2] + data[3] + data[4] - 4*(data[0])
    return laplacevalue

def minus(A, B): #takes original image and removes lightest parts of image to enhance darker areas - sharpen
    width, height = img.size
    copyimg = img.copy()
    pixels = copyimg.load()
    pixelsA = A.load()
    pixelsB = B.load()
    for i in range(0, width):
        for j in range(0, height):
            pixels[i,j] = pixelsA[i,j] - pixelsB[i,j]
    return copyimg


img = open(sys.argv)
img.show()
edges = filter(img, laplace)
minus(img, edges).show()
edges.show()