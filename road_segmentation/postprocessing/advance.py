from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from imgaug import augmenters as iaa
import gzip
import os
import sys
import urllib
import matplotlib.image as mpimg
from PIL import Image
import code
import tensorflow.python.platform
import numpy
import tensorflow as tf

IMG_PATCH_SIZE = 16
TRAIN_SIEZ = 1
TRAIN_WIDTH = 400
BATCH_SIZE = 16
EPOCHS = 5000
RECORDING_STEP = 10000
SEED = 66478
RESTORE_MODEL = False
PREDICTION_SIZE = 50
num_epochs = EPOCHS
PIXEL_DEPTH = 255
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', '/tmp/mnist',
                           """Directory where to write event logs """
                           """and checkpoint.""")
HIDDEN_SIZE = 256
def extract_labels(filename, num_images):
    """Extract the labels into a 1-hot matrix [image index, label index]."""
    gt_imgs = []
    for i in range(1, num_images + 1):
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            print('Loading ' + image_filename)
            img = mpimg.imread(image_filename)
            gt_imgs.append(img)
        else:
            print('File ' + image_filename + ' does not exist')

    num_images = len(gt_imgs)
    gt_patches = [img_crop(gt_imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE) for i in range(num_images)]
    data = numpy.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])
    labels = numpy.asarray([value_to_class(numpy.mean(data[i])) for i in range(len(data))])

    # Convert to dense 1-hot representation.
    return labels.astype(numpy.float32)


def extract_predictions(filename, num_images):
    """Extract the labels into a 1-hot matrix [image index, label index]."""
    gt_imgs = []
    for i in range(1, num_images + 1):
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            print('Loading ' + image_filename)
            img = mpimg.imread(image_filename)
            gt_imgs.append(img)
        else:
            print('File ' + image_filename + ' does not exist')

    num_images = len(gt_imgs)
    gt_patches = [img_crop(gt_imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE) for i in range(num_images)]
    data = numpy.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])
    labels = numpy.asarray([numpy.mean(data[i]) for i in range(len(data))])
    # Convert to dense 1-hot representation.
    return labels.astype(numpy.float32)

def extract_prediction(img):

    gt_imgs = []
    gt_imgs.append(img)
    num_images = len(gt_imgs)
    gt_patches = [img_crop(gt_imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE) for i in range(num_images)]
    data = numpy.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])
    labels = numpy.asarray([numpy.mean(data[i]) for i in range(len(data))])
    # Convert to dense 1-hot representation.
    return labels.astype(numpy.float32)

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


def value_to_class(v):
    foreground_threshold = 0.4  # percentage of pixels > 1 required to assign a foreground label to a patch
    df = numpy.sum(v)
    if df > foreground_threshold:
        return [0, 1]  # Road
    else:
        return [1, 0]  # Ground


def condition(row, col):
    if row > -1 and col > -1 and row < int(TRAIN_WIDTH / IMG_PATCH_SIZE) and col < int(TRAIN_WIDTH / IMG_PATCH_SIZE):
        return True
    else:
        return False


def error_rate(predictions, labels):
    """Return the error rate based on dense predictions and 1-hot labels."""
    return 100.0 - (
        100.0 *
        numpy.sum(numpy.argmax(predictions, 1) == numpy.argmax(labels, 1)) /
        predictions.shape[0])

def label_to_img(imgwidth, imgheight, w, h, labels):

    array_labels = numpy.zeros([imgwidth, imgheight])
    idx = 0

    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            array_labels[j:j+w, i:i+h] = labels[idx]
            '''
            original_label = origin[idx]
            if original_label == 0:
                if labels[idx][0] > labels[idx][1]: # Not Road -> Black
                    l = 0
                else:
                    l = 1  # Change It
                
                array_labels[j:j+w, i:i+h] = l
            else:
                #array_labels[j:j+w, i:i+h] = 1
                
                if labels[idx][0] > labels[idx][1]: # Not Road -> Black
                    l = 0
                else:
                    l = 1  # Change It
                
                array_labels[j:j+w, i:i+h] = l
            '''    
            idx = idx + 1
    return array_labels

def img_float_to_uint8(img):
    rimg = img - numpy.min(img)
    rimg = (rimg / numpy.max(rimg) * PIXEL_DEPTH).round().astype(numpy.uint8)
    return rimg

def convertToPNG(p_img):
    
    w = p_img.shape[0]
    h = p_img.shape[1]
    gt_img_3c = numpy.zeros((w, h, 3), dtype=numpy.uint8)
    gt_img8 = img_float_to_uint8(p_img)          
    gt_img_3c[:,:,0] = gt_img8
    gt_img_3c[:,:,1] = gt_img8
    gt_img_3c[:,:,2] = gt_img8
    return gt_img_3c

def get_prediction(img):
    gt_imgs = []
    gt_imgs.append(img)
    num_images = len(gt_imgs)
    gt_patches = [img_crop(gt_imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE) for i in range(num_images)]
    data = numpy.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])
    labels = numpy.asarray([numpy.mean(data[i]) for i in range(len(data))])
    prediction_labels = labels.reshape(38,38)
    size = prediction_labels.shape[0]-1
    connected_subpath = {}
    v_subpath = []
    h_subpath = []
    connected_subpath['v'] = v_subpath
    connected_subpath['h'] = h_subpath
    connection_list = []
    # For The Vertical One
    for i,col in enumerate(prediction_labels):
        left = 0
        right = 0
        count = 0
        col_subpath = []
        for j,cell in enumerate(col):
            if cell == 1:
                if count == 0:
                    left = j
                    right = j
                else:
                    right = j
                count+=1
            else:
                if count > 2:
                    # The SubPath
                    path = []
                    path.append(i) #ith column
                    path.append(left) #[left
                    path.append(right) #right]
                    path.append(count) #length
                    col_subpath.append(path)
                count = 0
                right = 0
                left = 0
        if count > 2:
            # The SubPath
            path = []
            path.append(i) #ith column
            path.append(left) #[left
            path.append(right) #right]
            path.append(count) #length
            col_subpath.append(path)

        connected_subpath['v'].append(col_subpath)

    for col_path in connected_subpath['v']:
        connected = False
        if len(col_path)>=2:
            col_path.sort(key=lambda x: -x[3])
            if col_path[0][3]>size/2 and col_path[1][3] > 3:
                gap = 0
                if col_path[0][1] > col_path[1][2]:
                    gap = col_path[0][1]- col_path[1][2]
                else: 
                    gap = col_path[1][1]-col_path[0][2]
                if gap < 10:
                    print('Connect!')
                    left = 0
                    right = 0
                    if col_path[0][1] > col_path[1][2]:
                        right = col_path[0][1]
                        left = col_path[1][2]
                    else:
                        right = col_path[1][1]
                        left = col_path[0][2]
                    connected = True
                    connection = []
                    connection.append(0) # 0 for vectical 1 for horizontal
                    connection.append(col_path[0][0]) # ith column
                    connection.append(left)
                    connection.append(right)
                    connection_list.append(connection)
        if connected:
            for path in col_path:
                print('Column '+str(path[0])+' : From '+str(path[1])+' To '+str(path[2])+' Of Length '+str(path[3]))


    # For The Horizontal One
    for i in range(prediction_labels.shape[1]):# ROW
        left = 0
        right = 0
        count = 0
        col_subpath = []
        for j in range(prediction_labels.shape[0]): #COL
            cell = prediction_labels[j][i] 
            if cell == 1:
                if count == 0:
                    left = j
                    right = j
                else:
                    right = j
                count+=1
            else:
                if count > 2:
                    # The SubPath
                    path = []
                    path.append(i) #ith row
                    path.append(left) #[left
                    path.append(right) #right]
                    path.append(count) #length
                    col_subpath.append(path)
                count = 0
                right = 0
                left = 0
        if count > 2:
            # The SubPath
            path = []
            path.append(i) #ith column
            path.append(left) #[left
            path.append(right) #right]
            path.append(count) #length
            col_subpath.append(path)

        connected_subpath['h'].append(col_subpath)

    for col_path in connected_subpath['h']:
        connected = False
        if len(col_path)>=2:
            col_path.sort(key=lambda x: -x[3])
            if col_path[0][3]>size/2 and col_path[1][3] > 3:
                gap = 0
                if col_path[0][1] > col_path[1][2]:
                    gap = col_path[0][1]- col_path[1][2]
                else: 
                    gap = col_path[1][1]-col_path[0][2]
                if gap < 10:
                    print('Connect!')
                    left = 0
                    right = 0
                    if col_path[0][1] > col_path[1][2]:
                        right = col_path[0][1]
                        left = col_path[1][2]
                    else:
                        right = col_path[1][1]
                        left = col_path[0][2]
                    connected = True
                    connection = []
                    connection.append(1) # 0 for vectical 1 for horizontal
                    connection.append(col_path[0][0]) # ith column
                    connection.append(left)
                    connection.append(right)
                    connection_list.append(connection)
        if connected:
            for path in col_path:
                print('ROW '+str(path[0])+' : From '+str(path[1])+' To '+str(path[2])+' Of Length '+str(path[3]))

    
    for connection in connection_list:
        wise = connection[1]
        left = connection[2]
        right = connection[3]
        if connection[0] == 1: #ROW WISE
            while left < right:
                left+=1
                prediction_labels[left][wise] = 1
        else:
            while left<right:
                left+=1
                prediction_labels[wise][left] = 1

    img_prediction = label_to_img(
        img.shape[0], img.shape[1], IMG_PATCH_SIZE, IMG_PATCH_SIZE, 
        prediction_labels.reshape(prediction_labels.shape[0]*prediction_labels.shape[1]))

    return img_prediction
    



                


    img_prediction = label_to_img(
        img.shape[0], img.shape[1], IMG_PATCH_SIZE, IMG_PATCH_SIZE, 
        prediction_labels.reshape(prediction_labels.shape[0]*prediction_labels.shape[1]))

    return img_prediction

p_dir = './pre_result/'
s_dir = './post_result/'
for i in range(1,51):
    imageid =  "%.3d"%i
    filename = 'prediction_'+str(imageid)+'.png'
    print(filename)
    img = mpimg.imread(p_dir+filename)
    p_img = get_prediction(img)
    img_3c = convertToPNG(p_img)
    Image.fromarray(img_3c).save(s_dir+filename)
    
