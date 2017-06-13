import gzip
import os
import sys
import urllib
import matplotlib.image as mpimg
from PIL import Image

import code

import tensorflow.python.platform
from skimage import io
import numpy
import tensorflow as tf
from imgaug import augmenters as iaa
import scipy.misc

IMG_PATCH_SIZE = 16

def extract_prediction(img):

    gt_imgs = []
    gt_imgs.append(img)
    num_images = len(gt_imgs)
    gt_patches = [img_crop(gt_imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE) for i in range(num_images)]
    data = numpy.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])
    data = numpy.asarray([numpy.mean(data[i]) for i in range(len(data))])
    labels = numpy.asarray([value_to_class(numpy.mean(data[i])) for i in range(len(data))])
    # Convert to dense 1-hot representation.
    return labels.astype(numpy.float32)

def value_to_class(v):
    foreground_threshold = 0.4 # percentage of pixels > 1 required to assign a foreground label to a patch
    df = numpy.sum(v)
    if df > foreground_threshold:
        return [0, 1]
    else:
        return [1, 0]

def img_crop(im, w, h):
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0, imgheight, h):
        for j in range(0, imgwidth, w):
            if is_2d:
                im_patch = im[j:j + w, i:i + h]
            else:
                im_patch = im[j:j + w, i:i + h, :]
            list_patches.append(im_patch)
    return list_patches

avg_t = 0
count = 0
for idx in range(1,101):

	filename = 'satImage_'+'{0:03}'.format(idx)+'.png'
	name = './predictions/'+filename
	p_img = io.imread(name, as_grey=True) #608*608
	p_labels = extract_prediction(p_img)

	name = './groundtruth/'+filename
	t_img = io.imread(name, as_grey=True) #608*608
	t_labels = extract_prediction(t_img)
	true_p = 0
	for idx,label in enumerate(p_labels):
		if(p_labels[idx][0]==t_labels[idx][0]):
				true_p +=1
	
	if true_p/p_labels.shape[0] > 0.88:
		count +=1
		print('Image : '+str(filename)+' : '+str(true_p/p_labels.shape[0]))
		filename = 'satImage_'+'{0:03}'.format(count)+'.png'
		scipy.misc.imsave('./post_train/' + filename, p_img)
		scipy.misc.imsave('./post_labels/' + filename, t_img)

	avg_t += true_p/p_labels.shape[0]
print('Average True Prediction : '+str(avg_t/100))
print('High Quality : '+str(count))

