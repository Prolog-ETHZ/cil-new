"""
Baseline for CIL project on road segmentation.
This simple baseline consits of a CNN with two convolutional+pooling layers with a soft-max loss
"""



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
from imgaug import augmenters as iaa
from imblearn.over_sampling import SMOTE
import pickle
from imblearn.over_sampling import RandomOverSampler 

NUM_CHANNELS = 3 # RGB images
PIXEL_DEPTH = 255
NUM_LABELS = 2
TRAINING_SIZE = 100
VALIDATION_SIZE = 5  # Size of the validation set.
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 16 # 64
NUM_EPOCHS = 5
RESTORE_MODEL = False # If True, restore existing model instead of training a new one
RECORDING_STEP = 1000
PREDICTION_SIZE = 50
# Set image patch size in pixels
# IMG_PATCH_SIZE should be a multiple of 4
# image size should be an integer multiple of this number!
IMG_PATCH_SIZE = 16
REFACTOR_PATCH_SIZE = IMG_PATCH_SIZE * 3
VARIANCE =  0.212212 #0.190137 #0.201966
PATCH_PER_IMAGE = 625
PRE_PROCESSED = False
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)

'''
tf.app.flags.DEFINE_string('train_dir', '/tmp/mnist',
                           """Directory where to write event logs """
                           """and checkpoint.""")
'''
FLAGS = tf.app.flags.FLAGS


# Extract patches from a given image
def img_crop(im, w, h):
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            if is_2d:
                im_patch = im[j:j+w, i:i+h]
            else:
                im_patch = im[j:j+w, i:i+h, :]
            list_patches.append(im_patch)
    return list_patches

def extract_data(filename, num_images):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    imgs = []
    for i in range(1, num_images+1):
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            print ('Loading ' + image_filename)
            img = mpimg.imread(image_filename)
            imgs.append(img)
        else:
            print ('File ' + image_filename + ' does not exist')

    num_images = len(imgs)
    IMG_WIDTH = imgs[0].shape[0]
    IMG_HEIGHT = imgs[0].shape[1]
    N_PATCHES_PER_IMAGE = (IMG_WIDTH/IMG_PATCH_SIZE)*(IMG_HEIGHT/IMG_PATCH_SIZE)

    img_patches = [img_crop(imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE) for i in range(num_images)]
    data = [img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))]

    return numpy.asarray(data)
        
# Assign a label to a patch v
def value_to_class(v):
    foreground_threshold = 0.25 # percentage of pixels > 1 required to assign a foreground label to a patch
    df = numpy.sum(v)
    if df > foreground_threshold:
        return [0, 1]
    else:
        return [1, 0]

# Extract label images
def extract_labels(filename, num_images):
    """Extract the labels into a 1-hot matrix [image index, label index]."""
    gt_imgs = []
    for i in range(1, num_images+1):
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            print ('Loading ' + image_filename)
            img = mpimg.imread(image_filename)
            gt_imgs.append(img)
        else:
            print ('File ' + image_filename + ' does not exist')

    num_images = len(gt_imgs)
    gt_patches = [img_crop(gt_imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE) for i in range(num_images)]
    data = numpy.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])
    labels = numpy.asarray([value_to_class(numpy.mean(data[i])) for i in range(len(data))])

    # Convert to dense 1-hot representation.
    return labels.astype(numpy.float32)


def error_rate(predictions, labels):
    """Return the error rate based on dense predictions and 1-hot labels."""
    return 100.0 - (
        100.0 *
        numpy.sum(numpy.argmax(predictions, 1) == numpy.argmax(labels, 1)) /
        predictions.shape[0]) 

# Write predictions from neural network to a file
def write_predictions_to_file(predictions, labels, filename):
    max_labels = numpy.argmax(labels, 1)
    max_predictions = numpy.argmax(predictions, 1)
    file = open(filename, "w")
    n = predictions.shape[0]
    for i in range(0, n):
        file.write(max_labels(i) + ' ' + max_predictions(i))
    file.close()

# Print predictions from neural network
def print_predictions(predictions, labels):
    max_labels = numpy.argmax(labels, 1)
    max_predictions = numpy.argmax(predictions, 1)
    print (str(max_labels) + ' ' + str(max_predictions))

# Convert array of labels to an image
def label_to_img(imgwidth, imgheight, w, h, labels):
    array_labels = numpy.zeros([imgwidth, imgheight])
    idx = 0
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            if labels[idx][0] > labels[idx][1]:
                l = 0
            else:
                l = 1
            array_labels[j:j+w, i:i+h] = 1-labels[idx][0]
            idx = idx + 1
    return array_labels

def img_float_to_uint8(img):
    rimg = img - numpy.min(img)
    rimg = (rimg / numpy.max(rimg) * PIXEL_DEPTH).round().astype(numpy.uint8)
    return rimg

def concatenate_images(img, gt_img):
    nChannels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if nChannels == 3:
        cimg = numpy.concatenate((img, gt_img), axis=1)
    else:
        gt_img_3c = numpy.zeros((w, h, 3), dtype=numpy.uint8)
        gt_img8 = img_float_to_uint8(gt_img)          
        gt_img_3c[:,:,0] = gt_img8
        gt_img_3c[:,:,1] = gt_img8
        gt_img_3c[:,:,2] = gt_img8
        img8 = img_float_to_uint8(img)
        cimg = numpy.concatenate((img8, gt_img_3c), axis=1)
    return cimg

def convertToPNG(p_img):
    
    w = p_img.shape[0]
    h = p_img.shape[1]
    gt_img_3c = numpy.zeros((w, h, 3), dtype=numpy.uint8)
    gt_img8 = img_float_to_uint8(p_img)          
    gt_img_3c[:,:,0] = gt_img8
    gt_img_3c[:,:,1] = gt_img8
    gt_img_3c[:,:,2] = gt_img8
    return gt_img_3c

def make_img_overlay(img, predicted_img):
    w = img.shape[0]
    h = img.shape[1]
    color_mask = numpy.zeros((w, h, 3), dtype=numpy.uint8)
    color_mask[:,:,0] = predicted_img*PIXEL_DEPTH

    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, 'RGB').convert("RGBA")
    overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img

def reformat(labels):
    new_labels = []
    for label in labels:
        if label == 1:
            new_labels.append([1,0])
        else:
            new_labels.append([0,1])

    return numpy.asarray(new_labels)

def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def read_object(filename):
    with open(filename,'rb') as output:
        data = pickle.load(output)
    return data


def main(argv=None):  # pylint: disable=unused-argument

    train_data = None
    train_labels = None
    num_epochs = NUM_EPOCHS
    if RESTORE_MODEL:
        TRAINING_SIZE = 1

    if not PRE_PROCESSED:

        data_dir = 'training/'
        train_data_filename = data_dir + 'images/'
        train_labels_filename = data_dir + 'groundtruth/' 
        
        # Extract it into numpy arrays.
        train_data = extract_data(train_data_filename, TRAINING_SIZE)
        train_labels = extract_labels(train_labels_filename, TRAINING_SIZE) #Ground = 1 Road = 0

        
        

        c0 = 0
        c1 = 0
        for i in range(len(train_labels)):
            if train_labels[i][0] == 1:
                c0 = c0 + 1
            else:
                c1 = c1 + 1
        print ('Number of data points per class: c0 = ' + str(c0) + ' c1 = ' + str(c1))
        diff = c0-c1
        print('Normalize the data .....')
        # === Feature Three : Pre-formalize the input ===
        avgs = [numpy.average(patch) for patch in train_data]
        train_data = numpy.asarray([[[[numpy.float32((val-avgs[patch[0]])/VARIANCE) for val in j] for j in i] for i in patch[1]] for patch in  enumerate(train_data)])
        print(train_data.shape)
        
        
        
        
        train_size = train_labels.shape[0]
        

        print('Refactor with neighbors ......')
        print(train_data.shape)
        print(train_labels.shape)
        refactored_train_data = list()
        for index,data in enumerate(train_data):

            img_idx = int(index / PATCH_PER_IMAGE)

            center = data
            up = data if (int((index-25)/PATCH_PER_IMAGE) != img_idx or index-25 <0) else train_data[index-25]
            down = data if  (int((index+25)/PATCH_PER_IMAGE) != img_idx) else train_data[index+25]
            left = data if (int((index-1)/PATCH_PER_IMAGE) != img_idx or index-1<0 ) else train_data[index-1]
            right = data if (int((index+1)/PATCH_PER_IMAGE) != img_idx) else train_data[index+1]
            up_left = data if  (int((index-25-1)/PATCH_PER_IMAGE) != img_idx or index-25-1 <0) else train_data[index-25-1]
            up_right = data if  (int((index-25+1)/PATCH_PER_IMAGE) != img_idx or index-25+1 <0) else train_data[index-25+1]
            down_left = data if  (int((index+25-1)/PATCH_PER_IMAGE) != img_idx) else train_data[index+25-1]
            down_right = data if  (int((index+25+1)/PATCH_PER_IMAGE) != img_idx) else train_data[index+25+1]
            
            
            mats = numpy.vstack([   numpy.hstack([up_left, up,up_right]), 
                                numpy.hstack([left, center,right]),
                                numpy.hstack([down_left, down,down_right])])
                
            refactored_train_data.append(mats)

        train_data = numpy.asarray(refactored_train_data)
        
        print('Form Global Data')
        global_train_data = list()
        for index,data in enumerate(train_data):

            img_idx = int(index / PATCH_PER_IMAGE)

            center = data
            up = data if (int((index-25*3)/PATCH_PER_IMAGE) != img_idx or index-25*3 <0) else train_data[index-25*3]
            down = data if  (int((index+25*3)/PATCH_PER_IMAGE) != img_idx) else train_data[index+25*3]
            left = data if (int((index-3)/PATCH_PER_IMAGE) != img_idx or index-3<0 ) else train_data[index-3]
            right = data if (int((index+3)/PATCH_PER_IMAGE) != img_idx) else train_data[index+3]
            up_left = data if  (int((index-25*3-3)/PATCH_PER_IMAGE) != img_idx or index-25*3-3 <0) else train_data[index-25*3-3]
            up_right = data if  (int((index-25*3+3)/PATCH_PER_IMAGE) != img_idx or index-25*3+3 <0) else train_data[index-25*3+3]
            down_left = data if  (int((index+25*3-3)/PATCH_PER_IMAGE) != img_idx) else train_data[index+25*3-3]
            down_right = data if  (int((index+25*3+3)/PATCH_PER_IMAGE) != img_idx) else train_data[index+25*3+3]
            
            
            mats = numpy.vstack([  numpy.hstack([up_left, up,up_right]), 
                                numpy.hstack([left, center,right]),
                                numpy.hstack([down_left, down,down_right])])
                
            global_train_data.append(mats)

        global_train_data = numpy.asarray(global_train_data)

        
        # New Feature : USE SMOTE To balance the data
        '''
		print ('Balancing training data...')
        sm = RandomOverSampler() #SMOTE(kind='regular')   # #
        # Change the shape to fit the method call
        train_data = train_data.reshape(train_data.shape[0],train_data.shape[1]*train_data.shape[2]*train_data.shape[3])
        global_train_data = global_train_data.reshape(global_train_data.shape[0],global_train_data.shape[1]*global_train_data.shape[2]*global_train_data.shape[3])
        train_labels =  train_labels[:,0]
        print(global_train_data.shape)
        
        print(train_data.shape)
        print(train_labels.shape)
        train_data, f_train_labels = sm.fit_sample(train_data, train_labels)
        global_train_data,train_labels = sm.fit_sample(global_train_data, train_labels)
        train_labels = f_train_labels
        print('reformat')
        train_data = train_data.reshape(train_data.shape[0],REFACTOR_PATCH_SIZE,REFACTOR_PATCH_SIZE,NUM_CHANNELS)
        global_train_data = global_train_data.reshape(global_train_data.shape[0],REFACTOR_PATCH_SIZE*3,REFACTOR_PATCH_SIZE*3,NUM_CHANNELS)
        train_labels = reformat(train_labels)
		'''
        print ('Balancing training data...')
        bal_index = [i for i in range(train_data.shape[0])]
        for i in range(diff):
            index = numpy.random.randint(train_data.shape[0])
            bal_index.append(index)
        print(len(bal_index))
        train_size = len(bal_index)


    else:

        train_data = read_object('./data/train_data.pkl')
        train_labels = read_object('./data/train_labels.pkl')
        print('Object Read')
        print('calculate new variance')
        var = numpy.std(train_data)
        print(var)

    print(train_data.shape)
    print(train_labels.shape)
    print(global_train_data.shape)
    train_size = train_labels.shape[0]
    c0 = 0
    c1 = 0
    for i in range(len(train_labels)):
        if train_labels[i][0] == 1:
            c0 = c0 + 1
        else:
            c1 = c1 + 1
    print ('Number of data points per class: c0 = ' + str(c0) + ' c1 = ' + str(c1))
    
    
    # This is where training samples and labels are fed to the graph.
    # These placeholder nodes will be fed a batch of training data at each
    # training step using the {feed_dict} argument to the Run() call below.
    train_data_node = tf.placeholder(
        tf.float32,
        shape=(BATCH_SIZE, REFACTOR_PATCH_SIZE, REFACTOR_PATCH_SIZE, NUM_CHANNELS)) # X_batch
    train_labels_node = tf.placeholder(tf.float32,
                                       shape=(BATCH_SIZE, NUM_LABELS)) # Y_batch
    global_data_node = tf.placeholder(
        tf.float32,
        shape=(BATCH_SIZE, REFACTOR_PATCH_SIZE*3, REFACTOR_PATCH_SIZE*3, NUM_CHANNELS)) # X_batch

    #train_all_data_node = tf.constant(train_data) # All_X
   

    # The variables below hold all the trainable weights. They are passed an
    # initial value which will be assigned when when we call:
    # {tf.initialize_all_variables().run()}

    # Define the sturcture of each layer of CNN

    # INPUT : N*48*48*3
    conv1_weights = tf.Variable(
        tf.truncated_normal([3, 3, NUM_CHANNELS, 32],  # Conv 3*3*64.
                            stddev=0.1,
                            seed=SEED))
    conv1_biases = tf.Variable(tf.zeros([32]))

    #INPUT: 48*48*64
    
    conv2_weights = tf.Variable(
        tf.truncated_normal([3, 3, 32, 32],  # Conv 3*3*64.
                            stddev=0.1,
                            seed=SEED))
    conv2_biases = tf.Variable(tf.zeros([32]))

    #INPUT:48*48*64
    #POOL:24*24*64

    #INPUT: 24*24*64
    conv3_weights = tf.Variable(
        tf.truncated_normal([3, 3, 32, 64],  # Conv 3*3*128.
                            stddev=0.1,
                            seed=SEED))
    conv3_biases = tf.Variable(tf.zeros([64]))

    #INPUT: 24*24*128
    conv4_weights = tf.Variable(
        tf.truncated_normal([3, 3, 64, 64],  # Conv 3*3*128.
                            stddev=0.1,
                            seed=SEED))
    conv4_biases = tf.Variable(tf.zeros([64]))

    #INPUT:24*24*128
    #POOL: 12*12*128

    #INPUT:12*12*128
    conv5_weights = tf.Variable(
        tf.truncated_normal([3, 3, 64, 128],  # Conv 3*3*256.
                            stddev=0.1,
                            seed=SEED))
    conv5_biases = tf.Variable(tf.zeros([128]))

    #INPUT:12*12*256
    conv6_weights = tf.Variable(
        tf.truncated_normal([3, 3, 128, 128],  # Conv 3*3*256.
                            stddev=0.1,
                            seed=SEED))
    conv6_biases = tf.Variable(tf.zeros([128]))

    #INPUT:12*12*256
    conv7_weights = tf.Variable(
        tf.truncated_normal([3, 3, 128, 128],  # Conv 3*3*256.
                            stddev=0.1,
                            seed=SEED))
    conv7_biases = tf.Variable(tf.zeros([128]))

    #INPUT:12*12*256
    #POOL: 6*6*256

    #INPUT:6*6*256
    conv8_weights = tf.Variable(
        tf.truncated_normal([3, 3, 128, 256],  # Conv 3*3*512.
                            stddev=0.1,
                            seed=SEED))
    conv8_biases = tf.Variable(tf.zeros([256]))

    #INPUT:6*6*512
    conv9_weights = tf.Variable(
        tf.truncated_normal([3, 3,256, 256],  # Conv 3*3*512.
                            stddev=0.1,
                            seed=SEED))
    conv9_biases = tf.Variable(tf.zeros([256]))

    #INPUT:6*6*512
    conv10_weights = tf.Variable(
        tf.truncated_normal([3, 3,256, 256],  # Conv 3*3*512.
                            stddev=0.1,
                            seed=SEED))
    conv10_biases = tf.Variable(tf.zeros([256]))

    #INPUT:6*6*512
    #POOL: 3*3*512

    #INPUT:3*3*512
    conv11_weights = tf.Variable(
        tf.truncated_normal([3, 3,256,256],  # Conv 3*3*512.
                            stddev=0.1,
                            seed=SEED))
    conv11_biases = tf.Variable(tf.zeros([256]))

    #INPUT:3*3*512
    conv12_weights = tf.Variable(
        tf.truncated_normal([3, 3,256, 256],  # Conv 3*3*512.
                            stddev=0.1,
                            seed=SEED))
    conv12_biases = tf.Variable(tf.zeros([256]))


    #INPUT:3*3*512
    conv13_weights = tf.Variable(
        tf.truncated_normal([3, 3,256, 256],  # Conv 3*3*512.
                            stddev=0.1,
                            seed=SEED))
    conv13_biases = tf.Variable(tf.zeros([256]))
    
    #INPUT:3*3*512
    #POOL:2*2*512
    #FLATEEN: 2048
    
    '''
    # Fully Connection 1
    fc1_weights = tf.Variable(  # fully connected, depth 512.
        tf.truncated_normal([int(3*3*512), 512],
                            stddev=0.1,
                            seed=SEED))
    fc1_biases = tf.Variable(tf.constant(0.1, shape=[512]))
    
    # Fully Connection 2
    fc2_weights = tf.Variable(  # fully connected, depth 128.
        tf.truncated_normal([512, 128],
                            stddev=0.1,
                            seed=SEED))
    fc2_biases = tf.Variable(tf.constant(0.1, shape=[128]))

    keep_prob = tf.placeholder(tf.float32)

    # Final Layer
    fc3_weights = tf.Variable(
        tf.truncated_normal([128, NUM_LABELS],
                            stddev=0.1,
                            seed=SEED))
    fc3_biases = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS]))
    '''
    # Change the Initializer

    conv1_weights = tf.get_variable('conv1_weights', 
        shape=(3, 3, NUM_CHANNELS, 32), initializer=tf.contrib.layers.xavier_initializer()) 

    conv2_weights = tf.get_variable('conv2_weights', 
        shape=(3, 3, 32, 32), initializer=tf.contrib.layers.xavier_initializer()) 

    conv3_weights = tf.get_variable('conv3_weights', 
        shape=(3, 3, 32, 64), initializer=tf.contrib.layers.xavier_initializer()) 

    conv4_weights = tf.get_variable('conv4_weights', 
        shape=(3, 3, 64, 64), initializer=tf.contrib.layers.xavier_initializer()) 

    conv5_weights = tf.get_variable('conv5_weights', 
        shape=(3, 3, 64, 128), initializer=tf.contrib.layers.xavier_initializer()) 

    conv6_weights = tf.get_variable('conv6_weights', 
        shape=(3, 3, 128, 128), initializer=tf.contrib.layers.xavier_initializer()) 

    conv7_weights = tf.get_variable('conv7_weights', 
        shape=(3, 3, 128, 128), initializer=tf.contrib.layers.xavier_initializer()) 

    conv8_weights = tf.get_variable('conv8_weights', 
        shape=(3, 3, 128, 256), initializer=tf.contrib.layers.xavier_initializer()) 

    conv9_weights = tf.get_variable('conv9_weights', 
        shape=(3, 3, 256, 256), initializer=tf.contrib.layers.xavier_initializer()) 

    conv10_weights = tf.get_variable('conv10_weights', 
        shape=(3, 3, 256, 256), initializer=tf.contrib.layers.xavier_initializer()) 

    conv11_weights = tf.get_variable('conv11_weights', 
        shape=(3, 3, 256, 256), initializer=tf.contrib.layers.xavier_initializer()) 

    conv12_weights = tf.get_variable('conv12_weights', 
        shape=(3, 3, 256, 256), initializer=tf.contrib.layers.xavier_initializer()) 

    conv13_weights = tf.get_variable('conv13_weights', 
        shape=(3, 3, 256, 256), initializer=tf.contrib.layers.xavier_initializer()) 


    keep_prob = tf.placeholder(tf.float32) 
    
    '''
    fc1_weights = tf.get_variable('fc1_weights', 
        shape=(int(3*3*512), 512), initializer=tf.contrib.layers.xavier_initializer()) 

    fc2_weights = tf.get_variable('fc2_weights', 
        shape=(512, 128), initializer=tf.contrib.layers.xavier_initializer()) 
    
    fc3_weights = tf.get_variable('fc3_weights', 
        shape=(128, NUM_LABELS), initializer=tf.contrib.layers.xavier_initializer()) 
    '''

    # ALEX NET
    
    # INPUT : N*144*144*3
    alex_conv1_weights = tf.Variable(
        tf.truncated_normal([11, 11, NUM_CHANNELS, 48],  # Conv 11*11*48.
                            stddev=0.1,
                            seed=SEED))
    alex_conv1_biases = tf.Variable(tf.zeros([48]))

    #INPUT:144*144*48
    #Pool :48*48*48

    # INPUT : N*48*48*48
    alex_conv2_weights = tf.Variable(
        tf.truncated_normal([5, 5, 48, 128],  # Conv 5*5*128.
                            stddev=0.1,
                            seed=SEED))
    alex_conv2_biases = tf.Variable(tf.zeros([128]))

    #INPUT:48*48*128
    #Pool :16*16*128

    # INPUT : N*16*16*128
    alex_conv3_weights = tf.Variable(
        tf.truncated_normal([3, 3, 128, 192],  # Conv 3*3*192.
                            stddev=0.1,
                            seed=SEED))
    alex_conv3_biases = tf.Variable(tf.zeros([192]))

    # INPUT : N*16*16*192
    alex_conv4_weights = tf.Variable(
        tf.truncated_normal([3, 3, 192, 192],  # Conv 3*3*192.
                            stddev=0.1,
                            seed=SEED))
    alex_conv4_biases = tf.Variable(tf.zeros([192]))

    # INPUT : N*16*16*192
    alex_conv5_weights = tf.Variable(
        tf.truncated_normal([3, 3, 192, 128],  # Conv 3*3*256.
                            stddev=0.1,
                            seed=SEED))
    alex_conv5_biases = tf.Variable(tf.zeros([128]))

    #INPUT:16*16*128
    #Pool :4*4*128

    alex_conv1_weights = tf.get_variable('alex_conv1_weights', 
        shape=(11,11, NUM_CHANNELS, 48), initializer=tf.contrib.layers.xavier_initializer()) 

    alex_conv2_weights = tf.get_variable('alex_conv2_weights', 
        shape=(5, 5, 48, 128), initializer=tf.contrib.layers.xavier_initializer()) 

    alex_conv3_weights = tf.get_variable('alex_conv3_weights', 
        shape=(3, 3, 128, 192), initializer=tf.contrib.layers.xavier_initializer()) 

    alex_conv4_weights = tf.get_variable('alex_conv4_weights', 
        shape=(3, 3, 192, 192), initializer=tf.contrib.layers.xavier_initializer()) 

    alex_conv5_weights = tf.get_variable('alex_conv5_weights', 
        shape=(3, 3, 192, 128), initializer=tf.contrib.layers.xavier_initializer()) 


    # Connection Layer

    # Fully Connection 1
    fc1_weights = tf.Variable(  # fully connected, depth 512.
        tf.truncated_normal([int(2*2*256+6*6*128), 512],
                            stddev=0.1,
                            seed=SEED))
    fc1_biases = tf.Variable(tf.constant(0.1, shape=[512]))
    
    # Fully Connection 2
    fc2_weights = tf.Variable(  # fully connected, depth 128.
        tf.truncated_normal([512, 2],
                            stddev=0.1,
                            seed=SEED))
    fc2_biases = tf.Variable(tf.constant(0.1, shape=[2]))

    fc1_weights = tf.get_variable('fc1_weights', 
        shape=(int(2*2*256+6*6*128), 512), initializer=tf.contrib.layers.xavier_initializer()) 

    fc2_weights = tf.get_variable('fc2_weights', 
        shape=(512, 2), initializer=tf.contrib.layers.xavier_initializer()) 


    total_parameters = numpy.sum([numpy.product([xi.value for xi in x.get_shape()]) for x in tf.all_variables()])
    print("Total Number Of Parameters :" +str(total_parameters))
    
    # Make an image summary for 4d tensor image with index idx
    def get_image_summary(img, idx = 0):
        V = tf.slice(img, (0, 0, 0, idx), (1, -1, -1, 1))
        img_w = img.get_shape().as_list()[1]
        img_h = img.get_shape().as_list()[2]
        min_value = tf.reduce_min(V)
        V = V - min_value
        max_value = tf.reduce_max(V)
        V = V / (max_value*PIXEL_DEPTH)
        V = tf.reshape(V, (img_w, img_h, 1))
        V = tf.transpose(V, (2, 0, 1))
        V = tf.reshape(V, (-1, img_w, img_h, 1))
        return V
    
    # Make an image summary for 3d tensor image with index idx
    def get_image_summary_3d(img):
        V = tf.slice(img, (0, 0, 0), (1, -1, -1))
        img_w = img.get_shape().as_list()[1]
        img_h = img.get_shape().as_list()[2]
        V = tf.reshape(V, (img_w, img_h, 1))
        V = tf.transpose(V, (2, 0, 1))
        V = tf.reshape(V, (-1, img_w, img_h, 1))
        return V

    # Get prediction for given input image 
    def get_prediction(img):

        p_data = numpy.asarray(img_crop(img, IMG_PATCH_SIZE, IMG_PATCH_SIZE))
        print(p_data.shape)
        refactored_data = list()
        for index,data in enumerate(p_data):

            center = data
            up = data if (index-38 <0) else p_data[index-38]
            down = data if(index+38>len(p_data)-1) else p_data[index+38]
            left = data if (index-1<0) else p_data[index-1]
            right = data if(index+1>len(p_data)-1) else p_data[index+1]
            up_left = data if  (index-38-1 <0) else p_data[index-38-1]
            up_right = data if  (index-38+1 > len(p_data)-1 ) else p_data[index-38+1]
            down_left = data if  (index+38-1 > len(p_data) -1) else p_data[index+38-1]
            down_right = data if  (index+38+1 > len(p_data)-1) else p_data[index+38+1]
            
            
            mats = numpy.vstack([numpy.hstack([up_left, up,up_right]), 
                            numpy.hstack([left, center,right]),
                            numpy.hstack([down_left, down,down_right])])
            refactored_data.append(mats)

        p_data = numpy.asarray(refactored_data)
        data_node = tf.constant(p_data)

        global_data = list()
        print(p_data.shape)
        for index,data in enumerate(p_data):

            center = data
            up = data if (index-38*3 <0) else p_data[index-38*3]
            down = data if(index+38*3>len(p_data)-3) else p_data[index+38*3]
            left = data if (index-3<0) else p_data[index-3]
            right = data if(index+3>len(p_data)-3) else p_data[index+3]
            up_left = data if  (index-38*3-3 <0) else p_data[index-38*3-3]
            up_right = data if  (index-38*3+3 > len(p_data)-3 ) else p_data[index-38*3+3]
            down_left = data if  (index+38*3-3 > len(p_data) -3) else p_data[index+38*3-3]
            down_right = data if  (index+38*3+3 > len(p_data)-3) else p_data[index+38*3+3]
           
            
            mats = numpy.vstack([numpy.hstack([up_left, up,up_right]), 
                            numpy.hstack([left, center,right]),
                            numpy.hstack([down_left, down,down_right])])
            global_data.append(mats)

        
        global_data = tf.constant(numpy.asarray(global_data))
        
        
        output = tf.nn.softmax(model(data_node,global_data))
        output_prediction = s.run(output,feed_dict={keep_prob:1.0})

        img_prediction = label_to_img(img.shape[0], img.shape[1], IMG_PATCH_SIZE, IMG_PATCH_SIZE, output_prediction)

        return img_prediction

    def get_prediction_for_train(img):

        p_data = numpy.asarray(img_crop(img, IMG_PATCH_SIZE, IMG_PATCH_SIZE))
        print(p_data.shape)
        refactored_data = list()
        for index,data in enumerate(p_data):

            center = data
            up = data if (index-25 <0) else p_data[index-25]
            down = data if(index+25>len(p_data)-1) else p_data[index+25]
            left = data if (index-1<0) else p_data[index-1]
            right = data if(index+1>len(p_data)-1) else p_data[index+1]
            up_left = data if  (index-25-1 <0) else p_data[index-25-1]
            up_right = data if  (index-25+1 > len(p_data)-1 ) else p_data[index-25+1]
            down_left = data if  (index+25-1 > len(p_data) -1) else p_data[index+25-1]
            down_right = data if  (index+25+1 > len(p_data)-1) else p_data[index+25+1]
            
            
            mats = numpy.vstack([numpy.hstack([up_left, up,up_right]), 
                            numpy.hstack([left, center,right]),
                            numpy.hstack([down_left, down,down_right])])
            refactored_data.append(mats)

        p_data = numpy.asarray(refactored_data)
        data_node = tf.constant(p_data)

        output = tf.nn.softmax(model(data_node))
        output_prediction = s.run(output,feed_dict={keep_prob:1.0})

        img_prediction = label_to_img(img.shape[0], img.shape[1], IMG_PATCH_SIZE, IMG_PATCH_SIZE, output_prediction)

        return img_prediction

    # Get a concatenation of the prediction and groundtruth for given input file
    def get_prediction_with_groundtruth(filename, image_idx):

        imageid = "satImage_%.3d" % image_idx
        image_filename = filename + imageid + ".png"
        img = mpimg.imread(image_filename)

        img_prediction = get_prediction(img)
        cimg = concatenate_images(img, img_prediction)

        return cimg

    # Get prediction overlaid on the original image for given input file
    def get_prediction_with_overlay(filename, image_idx):

        imageid = "satImage_%.3d" % image_idx
        image_filename = filename + imageid + ".png"
        img = mpimg.imread(image_filename)

        img_prediction = get_prediction(img)
        oimg = make_img_overlay(img, img_prediction)

        return oimg

    def max_out(inputs, num_units, axis=None):
        
        shape = inputs.get_shape().as_list()
        if shape[0] is None:
            shape[0] = -1
        if axis is None:  # Assume that channel is the last dimension
            axis = -1
        num_channels = shape[axis]
        if num_channels % num_units:
            raise ValueError('number of features({}) is not '
                         'a multiple of num_units({})'.format(num_channels, num_units))
        shape[axis] = num_units
        shape += [num_channels // num_units]
        outputs = tf.reduce_max(tf.reshape(inputs, shape), -1, keep_dims=False)
        return outputs


    # We will replicate the model structure for the training subgraph, as well
    # as the evaluation subgraphs, while sharing the trainable parameters.
    # Define the layer forward computation
    def model(data, global_data,train=False):
        """The Model definition."""
        # 2D convolution, with 'SAME' padding (i.e. the output feature map has
        # the same size as the input). Note that {strides} is a 4D array whose
        # shape matches the data layout: [image index, y, x, depth].
        # VGG-NET
        # =============== 1st Layer ==============
        # 3*3*64 Conv
        conv1 = tf.nn.conv2d(data,
                            conv1_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        # Relu the Conv
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

        # =============== 2nd Layer ==============
        # 3*3*64 Conv
        conv2 = tf.nn.conv2d(relu1,
                            conv2_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        # Relu the Conv
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

        # Max pooling. The kernel size spec {ksize} also follows the layout of
        # the data. Here we have a pooling window of 2, and a stride of 2.
        pool1 = tf.nn.max_pool(relu2,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME')

        # =============== 3rd Layer ==============
        # 3*3*128 Conv
        conv3 = tf.nn.conv2d(pool1,
                            conv3_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        relu3 = tf.nn.relu(tf.nn.bias_add(conv3, conv3_biases))

        # =============== 4th Layer ==============
        # 3*3*128 Conv
        conv4 = tf.nn.conv2d(relu3,
                            conv4_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        # Relu the Conv
        relu4 = tf.nn.relu(tf.nn.bias_add(conv4, conv4_biases))

        # Max pooling. The kernel size spec {ksize} also follows the layout of
        # the data. Here we have a pooling window of 2, and a stride of 2.
        pool2 = tf.nn.max_pool(relu4,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME')

        # =============== 5th Layer ==============
        # 3*3*256 Conv
        conv5 = tf.nn.conv2d(pool2,
                            conv5_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        relu5 = tf.nn.relu(tf.nn.bias_add(conv5, conv5_biases))

        # =============== 6th Layer ==============
        # 3*3*256 Conv
        conv6 = tf.nn.conv2d(relu5,
                            conv6_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        relu6 = tf.nn.relu(tf.nn.bias_add(conv6, conv6_biases))

        # =============== 7th Layer ==============
        # 3*3*256 Conv
        conv7 = tf.nn.conv2d(relu6,
                            conv7_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        relu7 = tf.nn.relu(tf.nn.bias_add(conv7, conv7_biases))

        # Max pooling. The kernel size spec {ksize} also follows the layout of
        # the data. Here we have a pooling window of 2, and a stride of 2.
        pool3 = tf.nn.max_pool(relu7,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME')

        # =============== 8th Layer ==============
        # 3*3*512 Conv
        conv8 = tf.nn.conv2d(pool3,
                            conv8_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        relu8 = tf.nn.relu(tf.nn.bias_add(conv8, conv8_biases))

        # =============== 9th Layer ==============
        # 3*3*512 Conv
        conv9 = tf.nn.conv2d(relu8,
                            conv9_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        relu9 = tf.nn.relu(tf.nn.bias_add(conv9, conv9_biases))

        # =============== 10th Layer ==============
        # 3*3*512 Conv
        conv10 = tf.nn.conv2d(relu9,
                            conv10_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        relu10 = tf.nn.relu(tf.nn.bias_add(conv10, conv10_biases))

        # Max pooling. The kernel size spec {ksize} also follows the layout of
        # the data. Here we have a pooling window of 2, and a stride of 2.
        pool4 = tf.nn.max_pool(relu10,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME')

        # =============== 11th Layer ==============
        # 3*3*512 Conv
        conv11 = tf.nn.conv2d(pool4,
                            conv11_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        relu11 = tf.nn.relu(tf.nn.bias_add(conv11, conv11_biases))

        # =============== 12th Layer ==============
        # 3*3*512 Conv
        conv12 = tf.nn.conv2d(relu11,
                            conv12_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        relu12 = tf.nn.relu(tf.nn.bias_add(conv12, conv12_biases))

        # =============== 13th Layer ==============
        # 3*3*512 Conv
        conv13 = tf.nn.conv2d(relu12,
                            conv13_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        relu13 = tf.nn.relu(tf.nn.bias_add(conv13, conv13_biases))

        # Max pooling. The kernel size spec {ksize} also follows the layout of
        # the data. Here we have a pooling window of 2, and a stride of 2.
        pool5 = tf.nn.max_pool(relu13,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME')


        #ALEX NET

        # =============== First Round ==============
        # 11*11*96 Conv
        alex_conv1 = tf.nn.conv2d(global_data,
                            alex_conv1_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        # Relu the Conv
        alex_relu1 = tf.nn.relu(tf.nn.bias_add(alex_conv1, alex_conv1_biases))
        # Max pooling. The kernel size spec {ksize} also follows the layout of
        # the data. Here we have a pooling window of 2, and a stride of 2.
        alex_pool1 = tf.nn.max_pool(alex_relu1,
                              ksize=[1, 3, 3, 1],
                              strides=[1, 3, 3, 1],
                              padding='SAME')

        # =============== Second Round ==============
        # 5*5*256 Conv
        alex_conv2 = tf.nn.conv2d(alex_pool1,
                            alex_conv2_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        # Relu the Conv
        alex_relu2 = tf.nn.relu(tf.nn.bias_add(alex_conv2, alex_conv2_biases))
        # Max pooling. The kernel size spec {ksize} also follows the layout of
        # the data. Here we have a pooling window of 2, and a stride of 2.
        alex_pool2 = tf.nn.max_pool(alex_relu2,
                              ksize=[1, 3, 3, 1],
                              strides=[1, 3, 3, 1],
                              padding='SAME')

        # =============== Third Round ==============
        # 3*3*384 Conv
        alex_conv3 = tf.nn.conv2d(alex_pool2,
                            alex_conv3_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        # Relu the Conv
        alex_relu3 = tf.nn.relu(tf.nn.bias_add(alex_conv3, alex_conv3_biases))

        # =============== Fourth Round ==============
        # 3*3*384 Conv
        alex_conv4 = tf.nn.conv2d(alex_relu3,
                            alex_conv4_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        # Relu the Conv
        alex_relu4 = tf.nn.relu(tf.nn.bias_add(alex_conv4, alex_conv4_biases))

        # =============== Second Round ==============
        # 5*5*256 Conv
        alex_conv5 = tf.nn.conv2d(alex_relu4,
                            alex_conv5_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        # Relu the Conv
        alex_relu5 = tf.nn.relu(tf.nn.bias_add(alex_conv5, alex_conv5_biases))
        # Max pooling. The kernel size spec {ksize} also follows the layout of
        # the data. Here we have a pooling window of 2, and a stride of 2.
        alex_pool5 = tf.nn.max_pool(alex_relu5,
                              ksize=[1, 3, 3, 1],
                              strides=[1, 3, 3, 1],
                              padding='SAME')

        # Fully Connection
        pool_shape1 = pool5.get_shape().as_list()
        
        reshape1 = tf.reshape(
            pool5,
            [pool_shape1[0], pool_shape1[1] * pool_shape1[2] * pool_shape1[3]])

        pool_shape2 = alex_pool5.get_shape().as_list()
        reshape2 = tf.reshape(
            alex_pool5,
            [pool_shape2[0], pool_shape2[1] * pool_shape2[2] * pool_shape2[3]])
        
        
        reshape = tf.concat([reshape1, reshape2], 1)
        
        hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
        hidden_drop = tf.nn.dropout(hidden, keep_prob)
        out = tf.matmul(hidden_drop, fc2_weights) + fc2_biases

        

        
        '''
        # =============== Final Stage ==============

        # Reshape the feature map cuboid into a 2D matrix to feed it to the
        # fully connected layers.
        pool_shape = pool4.get_shape().as_list()
        reshape = tf.reshape(
            pool4,
            [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
        # Fully connected layer. Note that the '+' operation automatically
        # broadcasts the biases.
        hidden1 = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
        hidden2 = tf.nn.relu(tf.matmul(hidden1, fc2_weights) + fc2_biases)
        hidden_drop = tf.nn.dropout(hidden2, keep_prob)
        out = tf.matmul(hidden_drop, fc3_weights) + fc3_biases
        '''

        if train == True:
            summary_id = '_0'
            
        return out

    # define the loss function
    # Training computation: logits + cross-entropy loss.
    logits = model(train_data_node, global_data_node,True) # BATCH_SIZE*NUM_LABELS
    # print 'logits = ' + str(logits.get_shape()) + ' train_labels_node = ' + str(train_labels_node.get_shape())
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=train_labels_node))

    # Visualize the Tensorflow

    tf.summary.scalar('loss', loss)

    all_params_node = [conv1_weights, conv1_biases, conv2_weights, conv2_biases, fc1_weights, fc1_biases, fc2_weights, fc2_biases]
    all_params_names = ['conv1_weights', 'conv1_biases', 'conv2_weights', 'conv2_biases', 'fc1_weights', 'fc1_biases', 'fc2_weights', 'fc2_biases']
    all_grads_node = tf.gradients(loss, all_params_node)
    all_grad_norms_node = []
    for i in range(0, len(all_grads_node)):
        norm_grad_i = tf.global_norm([all_grads_node[i]])
        all_grad_norms_node.append(norm_grad_i)
        tf.summary.scalar(all_params_names[i], norm_grad_i)
    
    # L2 regularization for the fully connected parameters.
    regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) +
                    tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases))
    # Add the regularization term to the loss.
    # Loss = loss function + regularizer
    loss += 5e-4 * regularizers 

    # Optimizer: set up a variable that's incremented once per batch and
    # controls the learning rate decay.
    batch = tf.Variable(0)
    # Decay once per epoch, using an exponential schedule starting at 0.01.
    learning_rate = tf.train.exponential_decay(
        0.009,                # Base learning rate.
        batch * BATCH_SIZE,  # Current index into the dataset.
        train_size*3 ,          # Decay step.
        0.95,                # Decay rate.
        staircase=True)
    tf.summary.merge_all(learning_rate)
    
    # Use simple momentum for the optimization.
    optimizer = tf.train.MomentumOptimizer(learning_rate,
                                           0.0).minimize(loss,
                                                         global_step=batch)

    # Predictions for the minibatch, validation set and test set.
    train_prediction = tf.nn.softmax(logits)
    # We'll compute them only once in a while by calling their {eval()} method.
    

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    # Create a local session to run this computation.
    with tf.Session() as s:

        s = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        if RESTORE_MODEL:
            # Restore variables from disk.
            saver.restore(s, "./model/model.ckpt")
            print("Model restored.") 

        else:
            # Run all the initializers to prepare the trainable parameters.
            tf.initialize_all_variables().run()

            # Build the summary operation based on the TF collection of Summaries.
            '''
            summary_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(FLAGS.train_dir,
                                                    graph_def=s.graph_def)
            '''
            print ('Initialized!')
            # Loop through training steps.
            print ('Total number of iterations = ' + str(int(num_epochs * train_size / BATCH_SIZE)))

            training_indices = range(train_size)
            iteration = 0
            for iepoch in range(num_epochs):
                print(str(iteration)+" iteration")
                iteration+=1
                # Permute training indices
                perm_indices = numpy.random.permutation(bal_index)

                for step in range (int(train_size / BATCH_SIZE)):

                    offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
                    batch_indices = perm_indices[offset:(offset + BATCH_SIZE)]

                    # Compute the offset of the current minibatch in the data.
                    # Note that we could use better randomization across epochs.
                    batch_data = train_data[batch_indices, :, :, :]
                    batch_labels = train_labels[batch_indices]
                    batch_global_data = global_train_data[batch_indices, :, :, :]
                    # This dictionary maps the batch data (as a numpy array) to the
                    # node in the graph is should be fed to.

                    # === Feature One : Rotate Image Randomly ===
                    angel = numpy.random.randint(360)
                    seq = iaa.Sequential([
                        iaa.Affine(
                            rotate=(angel, angel), # rotate randomly
                        ), 
                    ])
                    batch_data = seq.augment_images(batch_data)
                    batch_global_data = seq.augment_images(batch_global_data)
                    # === Feature One : Rotate Image Randomly ===

                    

                    if step % RECORDING_STEP == 0:

                        # === Feature Two : Add ===

                        feed_dict={train_data_node:batch_data,global_data_node:batch_global_data 
                                    ,train_labels_node:batch_labels,keep_prob:1.0}
                        _, l, lr, predictions = s.run(
                            [optimizer, loss, learning_rate, train_prediction],
                            feed_dict=feed_dict)
                        #summary_str = s.run(summary_op, feed_dict=feed_dict)
                        #summary_writer.add_summary(summary_str, step)
                        #summary_writer.flush()

                        # print_predictions(predictions, batch_labels)

                        print ('Epoch %.2f' % (float(step) * BATCH_SIZE / train_size))
                        print ('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
                        print ('Minibatch error: %.1f%%' % error_rate(predictions,
                                                                     batch_labels))

                        sys.stdout.flush()
                    else:
                        # Run the graph and fetch some of the nodes.

                        feed_dict = {train_data_node: batch_data, global_data_node:batch_global_data ,
                                 train_labels_node: batch_labels,
                                 keep_prob:0.5}

                        _, l, lr, predictions = s.run(
                            [optimizer, loss, learning_rate, train_prediction],
                            feed_dict=feed_dict)

                # Save the variables to disk. Check if the model is better if not, don't store
                
                

        #save_path = saver.save(s, "./model/model.ckpt")
        #print("Model saved in file: %s" % save_path)
        print ("Running prediction on training set")
        
        # Feature Four : Final Assesement on Training Data
        
        prediction_training_dir = "predictions_training/"
        if not os.path.isdir(prediction_training_dir):
            os.mkdir(prediction_training_dir)
        
        
        predict_data_filename = './test_set_images';
        for idx in range(1,PREDICTION_SIZE+1):
            name = predict_data_filename+'/test_'+str(idx)+'/test_'+str(idx)+'.png'
            img = mpimg.imread(name)
            print(name)
            avg = numpy.average(img)
            img = numpy.asarray([[[numpy.float32((val-avg)/VARIANCE) for val in i] for i in j ] for j in img ])
            
            p_img = get_prediction(img)
            img_3c = convertToPNG(p_img)
            Image.fromarray(img_3c).save(prediction_training_dir + "prediction_" + str(idx) + ".png")

        

if __name__ == '__main__':
    tf.app.run()
