from PIL import Image
import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np

for i in range(1,101):
	idx = "%.3d"%i
	image_name = './satImage_' +str(idx)+ '.png'
	image = Image.open(image_name)
	# Generate Contrast
	idx1 = "%.3d"%(i+100)
	gen1 = image
	gen1.save('./satImage_' +str(idx1)+ '.png')
	# Generate Noise
	idx2 = "%.3d"%(i+200)
	gen2 = image
	gen2.save('./satImage_' +str(idx2)+ '.png')