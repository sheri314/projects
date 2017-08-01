import sys
from PIL import Image

# define your flip function here
def flip(filename):
	width, height = img.size
	imgflipped = img.copy()
	originalpixels = img.load()	# 2D matrix of original image
	copypixels = imgflipped.load() # 2D matrix of copy image **used imgorig
	for i in range(0,width): # iterates over x values
		for j in range(0,height): #iterates over y values
			copypixels[i,j] = originalpixels[width-i-1,j]
	return imgflipped # returns the flipped image


if len(sys.argv)<=1:
	print "missing image filename"
	sys.exit(1)
filename = sys.argv[1]
img = Image.open(filename)
img = img.convert("L")
img.show()
flip(filename).show()
