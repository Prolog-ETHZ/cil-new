from PIL import Image
import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np

def change_contrast(img, level):

    factor = (259 * (level + 255)) / (255 * (259 - level))
    def contrast(c):
        return 128 + factor * (c - 128)
    return img.point(contrast)

def aug_img(img):

	seq = iaa.Sequential([
		iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
		#iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),                
	])
	images_aug = seq.augment_images(np.array(img))
	return Image.fromarray(images_aug, 'RGB')










for i in range(1,51):

	idx = i
	image_name = './test_' +str(idx)+'/test_'+str(idx)+ '.png'
	image = Image.open(image_name)
	# Generate Contrast
	idx1 = "%.3d"%(i+100)
	gen1 = change_contrast(image,50)
	gen1.save(image_name)

  
