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

NUM_CHANNELS = 3 # RGB images
PIXEL_DEPTH = 255
NUM_LABELS = 2
TRAINING_SIZE = 10
VALIDATION_SIZE = 5  # Size of the validation set.
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 16 # 64
NUM_EPOCHS = 1
RESTORE_MODEL = False # If True, restore existing model instead of training a new one
RECORDING_STEP = 1000
PREDICTION_SIZE = 50
# Set image patch size in pixels
# IMG_PATCH_SIZE should be a multiple of 4
# image size should be an integer multiple of this number!
IMG_PATCH_SIZE = 16
REFACTOR_PATCH_SIZE = IMG_PATCH_SIZE * 3
VARIANCE = 0.161416 #0.190137
PATCH_PER_IMAGE = 625

tf.app.flags.DEFINE_string('train_dir', '/tmp/mnist',
                           """Directory where to write event logs """
                           """and checkpoint.""")
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
            array_labels[j:j+w, i:i+h] = labels[idx][0]
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


def main(argv=None):  # pylint: disable=unused-argument

    data_dir = 'training/'
    train_data_filename = data_dir + 'images/'
    train_labels_filename = data_dir + 'groundtruth/' 
    
    # Extract it into numpy arrays.
    train_data = extract_data(train_data_filename, TRAINING_SIZE)
    train_labels = extract_labels(train_labels_filename, TRAINING_SIZE) #Ground = 1 Road = 0

    
    num_epochs = NUM_EPOCHS

    c0 = 0
    c1 = 0
    for i in range(len(train_labels)):
        if train_labels[i][0] == 1:
            c0 = c0 + 1
        else:
            c1 = c1 + 1
    print ('Number of data points per class: c0 = ' + str(c0) + ' c1 = ' + str(c1))
    '''
    print ('Balancing training data...')
    min_c = min(c0, c1)
    idx0 = [i for i, j in enumerate(train_labels) if j[0] == 1]
    idx1 = [i for i, j in enumerate(train_labels) if j[1] == 1]
    new_indices = idx0[0:min_c] + idx1[0:min_c]
    print (len(new_indices))
    print (train_data.shape)
    train_data = train_data[new_indices,:,:,:]
    train_labels = train_labels[new_indices]

    print(train_data.shape)  #sample_size * 16 * 16 *RGB
    print(train_labels[0][1])  # probability of [0]=>Ground [1]=>Round
    '''
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
   
    
    
    
    print(train_data.shape)
    print(train_labels.shape)
    
   
    
    # This is where training samples and labels are fed to the graph.
    # These placeholder nodes will be fed a batch of training data at each
    # training step using the {feed_dict} argument to the Run() call below.
    train_data_node = tf.placeholder(
        tf.float32,
        shape=(BATCH_SIZE, REFACTOR_PATCH_SIZE, REFACTOR_PATCH_SIZE, NUM_CHANNELS)) # X_batch
    train_labels_node = tf.placeholder(tf.float32,
                                       shape=(BATCH_SIZE, NUM_LABELS)) # Y_batch

    train_all_data_node = tf.constant(train_data) # All_X
   

    # The variables below hold all the trainable weights. They are passed an
    # initial value which will be assigned when when we call:
    # {tf.initialize_all_variables().run()}

    # Define the sturcture of each layer of CNN
    '''
    conv1_weights = tf.Variable(
        tf.truncated_normal([5, 5, NUM_CHANNELS, 32],  # 5x5 filter, depth 32.
                            stddev=0.1,
                            seed=SEED))
    conv1_biases = tf.Variable(tf.zeros([32]))
    conv2_weights = tf.Variable(
        tf.truncated_normal([5, 5, 32, 64],
                            stddev=0.1,
                            seed=SEED))
    conv2_biases = tf.Variable(tf.constant(0.1, shape=[64]))
    
    fc1_weights = tf.Variable(  # fully connected, depth 512.
        tf.truncated_normal([int(REFACTOR_PATCH_SIZE/ 4 * REFACTOR_PATCH_SIZE / 4 * 64), 512],
                            stddev=0.1,
                            seed=SEED))
    fc1_biases = tf.Variable(tf.constant(0.1, shape=[512]))
    keep_prob = tf.placeholder(tf.float32)
    fc2_weights = tf.Variable(
        tf.truncated_normal([512, NUM_LABELS],
                            stddev=0.1,
                            seed=SEED))
    fc2_biases = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS]))
    '''

    # INPUT : N*48*48*3
    conv1_weights = tf.Variable(
        tf.truncated_normal([3, 3, NUM_CHANNELS, 64],  # Conv 3*3*64.
                            stddev=0.1,
                            seed=SEED))
    conv1_biases = tf.Variable(tf.zeros([64]))
    
    # INPUT:48*48*64
    # Pool : 24*24*64

    conv2_weights = tf.Variable(
        tf.truncated_normal([3, 3, 64, 128],  # Conv 3*3*128.
                            stddev=0.1,
                            seed=SEED))
    conv2_biases = tf.Variable(tf.zeros([128]))

    # INPUT:24*24*128
    # Pool:12*12*128

    conv3_weights = tf.Variable(
        tf.truncated_normal([3, 3, 128, 256],  # Conv 3*3*256.
                            stddev=0.1,
                            seed=SEED))
    conv3_biases = tf.Variable(tf.zeros([256]))

    #INPUT:12*12*256
    #Pool : 6*6*256

    conv4_weights = tf.Variable(
        tf.truncated_normal([3, 3, 256, 512],  # Conv 3*3*512.
                            stddev=0.1,
                            seed=SEED))
    conv4_biases = tf.Variable(tf.zeros([512]))

    #INPUT:6*6*512
    #Pool :3*3*512

    # Fully Connection
    fc1_weights = tf.Variable(  # fully connected, depth 512.
        tf.truncated_normal([int(3*3*512), 512],
                            stddev=0.1,
                            seed=SEED))
    fc1_biases = tf.Variable(tf.constant(0.1, shape=[512]))
    keep_prob = tf.placeholder(tf.float32)

    # Final Layer
    fc2_weights = tf.Variable(
        tf.truncated_normal([512, NUM_LABELS],
                            stddev=0.1,
                            seed=SEED))
    fc2_biases = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS]))

    # Change the Initializer

    conv1_weights = tf.get_variable('conv1_weights', 
        shape=(3, 3, NUM_CHANNELS, 64), initializer=tf.contrib.layers.xavier_initializer()) 

    conv2_weights = tf.get_variable('conv2_weights', 
        shape=(3, 3, 64, 128), initializer=tf.contrib.layers.xavier_initializer()) 

    conv3_weights = tf.get_variable('conv3_weights', 
        shape=(3, 3, 128, 256), initializer=tf.contrib.layers.xavier_initializer()) 

    conv4_weights = tf.get_variable('conv4_weights', 
        shape=(3, 3, 256, 512), initializer=tf.contrib.layers.xavier_initializer()) 

    fc1_weights = tf.get_variable('fc1_weights', 
        shape=(int(3*3*512), 512), initializer=tf.contrib.layers.xavier_initializer()) 
    
    fc2_weights = tf.get_variable('fc2_weights', 
        shape=(512, NUM_LABELS), initializer=tf.contrib.layers.xavier_initializer()) 

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
    def model(data, train=False):
        """The Model definition."""
        # 2D convolution, with 'SAME' padding (i.e. the output feature map has
        # the same size as the input). Note that {strides} is a 4D array whose
        # shape matches the data layout: [image index, y, x, depth].
        '''
        conv = tf.nn.conv2d(data,
                            conv1_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        # Bias and rectified linear non-linearity.
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
        # Max pooling. The kernel size spec {ksize} also follows the layout of
        # the data. Here we have a pooling window of 2, and a stride of 2.
        pool = tf.nn.max_pool(relu,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME')

        conv2 = tf.nn.conv2d(pool,
                            conv2_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
        pool2 = tf.nn.max_pool(relu2,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME')

        


        # Reshape the feature map cuboid into a 2D matrix to feed it to the
        # fully connected layers.
        pool_shape = pool2.get_shape().as_list()
        reshape = tf.reshape(
            pool2,
            [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
        # Fully connected layer. Note that the '+' operation automatically
        # broadcasts the biases.
        hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
        # Add a 50% dropout during training only. Dropout also scales
        # activations such that no rescaling is needed at evaluation time.
        #if train:
        #    hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)
        hidden_drop = tf.nn.dropout(hidden, keep_prob)

        out = tf.matmul(hidden_drop, fc2_weights) + fc2_biases
        '''
        # =============== First Round ==============
        # 3*3*64 Conv
        conv1 = tf.nn.conv2d(data,
                            conv1_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        # Relu the Conv
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))
        # Max pooling. The kernel size spec {ksize} also follows the layout of
        # the data. Here we have a pooling window of 2, and a stride of 2.
        pool1 = tf.nn.max_pool(relu1,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME')


        # =============== Second Round ==============
        # 3*3*128 Conv
        conv2 = tf.nn.conv2d(pool1,
                            conv2_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
        pool2 = tf.nn.max_pool(relu2,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME')

        # =============== Third Round ==============
        # 3*3*256 Conv
        conv3 = tf.nn.conv2d(pool2,
                            conv3_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        relu3 = tf.nn.relu(tf.nn.bias_add(conv3, conv3_biases))
        pool3 = tf.nn.max_pool(relu3,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME')

        # =============== Fourth Round ==============
        # 3*3*512 Conv
        conv4 = tf.nn.conv2d(pool3,
                            conv4_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        relu4 = tf.nn.relu(tf.nn.bias_add(conv4, conv4_biases))
        pool4 = tf.nn.max_pool(relu4,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME')

        # =============== Final Stage ==============

        # Reshape the feature map cuboid into a 2D matrix to feed it to the
        # fully connected layers.
        pool_shape = pool4.get_shape().as_list()
        reshape = tf.reshape(
            pool4,
            [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
        # Fully connected layer. Note that the '+' operation automatically
        # broadcasts the biases.
        hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
        hidden_drop = tf.nn.dropout(hidden, keep_prob)
        out = tf.matmul(hidden_drop, fc2_weights) + fc2_biases

        if train == True:
            summary_id = '_0'
            '''
            s_data = get_image_summary(data)
            filter_summary0 = tf.summary.image('summary_data' + summary_id, s_data)
            s_conv = get_image_summary(conv)
            filter_summary2 = tf.summary.image('summary_conv' + summary_id, s_conv)
            s_pool = get_image_summary(pool)
            filter_summary3 = tf.summary.image('summary_pool' + summary_id, s_pool)
            s_conv2 = get_image_summary(conv2)
            filter_summary4 = tf.summary.image('summary_conv2' + summary_id, s_conv2)
            s_pool2 = get_image_summary(pool2)
            filter_summary5 = tf.summary.image('summary_pool2' + summary_id, s_pool2)
            '''
        return out

    # define the loss function
    # Training computation: logits + cross-entropy loss.
    logits = model(train_data_node, True) # BATCH_SIZE*NUM_LABELS
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
        0.015,                # Base learning rate.
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


        if RESTORE_MODEL:
            # Restore variables from disk.
            saver.restore(s, FLAGS.train_dir + "/model.ckpt")
            print("Model restored.")

        else:
            # Run all the initializers to prepare the trainable parameters.
            tf.initialize_all_variables().run()

            # Build the summary operation based on the TF collection of Summaries.
            summary_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(FLAGS.train_dir,
                                                    graph_def=s.graph_def)
            print ('Initialized!')
            # Loop through training steps.
            print ('Total number of iterations = ' + str(int(num_epochs * train_size / BATCH_SIZE)))

            training_indices = range(train_size)
            iteration = 0
            for iepoch in range(num_epochs):
                print(str(iteration)+" iteration")
                iteration+=1
                # Permute training indices
                perm_indices = numpy.random.permutation(training_indices)

                for step in range (int(train_size / BATCH_SIZE)):

                    offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
                    batch_indices = perm_indices[offset:(offset + BATCH_SIZE)]

                    # Compute the offset of the current minibatch in the data.
                    # Note that we could use better randomization across epochs.
                    batch_data = train_data[batch_indices, :, :, :]
                    batch_labels = train_labels[batch_indices]
                    
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
                    # === Feature One : Rotate Image Randomly ===

                    

                    if step % RECORDING_STEP == 0:

                        # === Feature Two : Add ===

                        feed_dict={train_data_node:batch_data,train_labels_node:batch_labels,keep_prob:1.0}
                        summary_str, _, l, lr, predictions = s.run(
                            [summary_op, optimizer, loss, learning_rate, train_prediction],
                            feed_dict=feed_dict)
                        #summary_str = s.run(summary_op, feed_dict=feed_dict)
                        summary_writer.add_summary(summary_str, step)
                        summary_writer.flush()

                        # print_predictions(predictions, batch_labels)

                        print ('Epoch %.2f' % (float(step) * BATCH_SIZE / train_size))
                        print ('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
                        print ('Minibatch error: %.1f%%' % error_rate(predictions,
                                                                     batch_labels))

                        sys.stdout.flush()
                    else:
                        # Run the graph and fetch some of the nodes.

                        feed_dict = {train_data_node: batch_data,
                                 train_labels_node: batch_labels,
                                 keep_prob:0.5}

                        _, l, lr, predictions = s.run(
                            [optimizer, loss, learning_rate, train_prediction],
                            feed_dict=feed_dict)

                # Save the variables to disk. Check if the model is better if not, don't store
                
                save_path = saver.save(s, FLAGS.train_dir + "/model.ckpt")
                print("Model saved in file: %s" % save_path)


        print ("Running prediction on training set")
        
        # Feature Four : Final Assesement on Training Data
        
        prediction_training_dir = "predictions_training/"
        if not os.path.isdir(prediction_training_dir):
            os.mkdir(prediction_training_dir)
        '''
        final_model = model(train_all_data_node)
        train_all_prediction = tf.nn.softmax(final_model)
        final_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=final_model, labels=train_labels))
        final_loss += 5e-4 * regularizers
        loss,predictions = s.run([final_loss,train_all_prediction],feed_dict={keep_prob:1})
        
        print ('error on trainning data : %.1f%%' % error_rate(predictions,train_labels))
        print ('loss on trainning data: %.3f' % loss)
        '''

        '''
        for i in range(1, TRAINING_SIZE+1):
            name = ''
            
            pimg = get_prediction_with_groundtruth(train_data_filename, i)
            Image.fromarray(pimg).save(prediction_training_dir + "prediction_" + str(i) + ".png")
            oimg = get_prediction_with_overlay(train_data_filename, i)
            oimg.save(prediction_training_dir + "overlay_" + str(i) + ".png")    
            '''
        
        
        predict_data_filename = './test_set_images';
        for i in range(1,PREDICTION_SIZE+1):
            name = predict_data_filename+'/test_'+str(i)+'/test_'+str(i)+'.png'
            img = mpimg.imread(name)
            print(name)
            avg = numpy.average(img)
            img = numpy.asarray([[[numpy.float32((val-avg)/VARIANCE) for val in i] for i in j ] for j in img ])
            
            p_img = get_prediction(img)
            img_3c = convertToPNG(p_img)
            Image.fromarray(img_3c).save(prediction_training_dir + "prediction_" + str(i) + ".png")
        
        train_data_filename = './training/images/';
        dir = './training/predictions/'
        if not os.path.isdir(dir):
            os.mkdir(dir)
        for i in range(1,TRAINING_SIZE+1):
            name = train_data_filename+'satImage_'+'{0:03}'.format(i)+'.png'
            img = mpimg.imread(name)
            print(name)
            avg = numpy.average(img)
            img = numpy.asarray([[[numpy.float32((val-avg)/VARIANCE) for val in i] for i in j ] for j in img ])
            
            p_img = get_prediction_for_train(img)
            img_3c = convertToPNG(p_img)
            Image.fromarray(img_3c).save(dir + name)

if __name__ == '__main__':
    tf.app.run()
