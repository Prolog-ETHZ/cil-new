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
TRAIN_SIEZ = 100
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

def label_to_img(origin,imgwidth, imgheight, w, h, labels):

    array_labels = numpy.zeros([imgwidth, imgheight])
    idx = 0

    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            
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
    #print(prediction_labels[0])
    p_data = []
    image = prediction_labels
    for i, _ in enumerate(prediction_labels):
        for j, _ in enumerate(prediction_labels[i]):
            # Here is the result x
            # print(image[i][j])
            data = []
            # Left Right Up Down for 5 step
            curr_left = image[i][j]
            curr_right = image[i][j]
            curr_up = image[i][j]
            curr_down = image[i][j]
            for step in range(1,4):
                
                # UP
                if condition(i+step,j):
                    data.append(image[i+step][j])
                    curr_up = image[i+step][j]
                else:
                    data.append(curr_up)
                # DOWN
                if condition(i-step,j):
                    data.append(image[i-step][j])
                    curr_down = image[i-step][j]
                else:
                    data.append(curr_down)
                
                #RIGHT
                if condition(i,j+step):
                    data.append(image[i][j+step])
                    curr_right = image[i][j+step]
                else:
                    data.append(curr_right)
                #LEFT
                if condition(i,j-step):
                    data.append(image[i][j-step])
                    curr_left = image[i][j-step]
                else:
                    data.append(curr_left)


            # Add Suroundings
            '''
            for row in [1, -1]:
                for col in [1, -1]:
                    data.append(image[i + row][j + col] if condition(i + row, j + col) else image[i][j])
                    data.append(image[i + 2 * row][j + col] if condition(i + 2 * row, j + col) else image[i][j])
                    data.append(image[i + row][j + 2 * col] if condition(i + row, j + 2 * col) else image[i][j])
                    data.append(image[i + 2 * row][j + 2 * col] if condition(i + 2 * row, j + 2 * col) else image[i][j])
            '''
            # Add Itself
            #data.append(image[i][j])
            p_data.append(numpy.asarray(data))
    p_data = numpy.asarray(p_data)
    #print(p_data.shape)
    data_node = tf.constant(p_data)
    output = tf.nn.softmax(model(data_node))
    output_prediction = s.run(output)
    img_prediction = label_to_img(prediction_labels.reshape(prediction_labels.shape[0]*prediction_labels.shape[1],-1),
        img.shape[0], img.shape[1], IMG_PATCH_SIZE, IMG_PATCH_SIZE, output_prediction)
    return img_prediction
        

# Load The Labels -> y
train_labels = extract_labels('./post_train_labels/', TRAIN_SIEZ)
#print(train_labels.shape)
# Form The Input -> x
predicted_labels = extract_predictions('./post_train_img/', TRAIN_SIEZ)

predicted_labels = predicted_labels.reshape(TRAIN_SIEZ, int(TRAIN_WIDTH / IMG_PATCH_SIZE),
                                            int(TRAIN_WIDTH / IMG_PATCH_SIZE))

train_dataset = []
for image in predicted_labels:
    for i, _ in enumerate(image):
        for j, _ in enumerate(image[i]):
            # Here is the result x
            # print(image[i][j])
            data = []
            
            # Left Right Up Down for 5 step
            curr_left = image[i][j]
            curr_right = image[i][j]
            curr_up = image[i][j]
            curr_down = image[i][j]
            for step in range(1,4):
                
                # UP
                if condition(i+step,j):
                    data.append(image[i+step][j])
                    curr_up = image[i+step][j]
                else:
                    data.append(curr_up)
                # DOWN
                if condition(i-step,j):
                    data.append(image[i-step][j])
                    curr_down = image[i-step][j]
                else:
                    data.append(curr_down)
                
                #RIGHT
                if condition(i,j+step):
                    data.append(image[i][j+step])
                    curr_right = image[i][j+step]
                else:
                    data.append(curr_right)
                #LEFT
                if condition(i,j-step):
                    data.append(image[i][j-step])
                    curr_left = image[i][j-step]
                else:
                    data.append(curr_left)


            # Add Suroundings
            '''
            for row in [1, -1]:
                for col in [1, -1]:
                    data.append(image[i + row][j + col] if condition(i + row, j + col) else image[i][j])
                    data.append(image[i + 2 * row][j + col] if condition(i + 2 * row, j + col) else image[i][j])
                    data.append(image[i + row][j + 2 * col] if condition(i + row, j + 2 * col) else image[i][j])
                    data.append(image[i + 2 * row][j + 2 * col] if condition(i + 2 * row, j + 2 * col) else image[i][j])
            # Add Itself
            '''
            #data.append(image[i][j])
            
            train_dataset.append(numpy.asarray(data))

train_dataset = numpy.asarray(train_dataset)
#print(train_dataset.shape)
c0 = 0
c1 = 0
for i in range(len(train_labels)):
    if train_labels[i][0] == 1:
        c0 = c0 + 1
    else:
        c1 = c1 + 1
print ('Number of data points per class: c0 = ' + str(c0) + ' c1 = ' + str(c1))
    
print ('Balancing training data...')
min_c = min(c0, c1)
idx0 = [i for i, j in enumerate(train_labels) if j[0] == 1]
idx1 = [i for i, j in enumerate(train_labels) if j[1] == 1]
new_indices = idx0[0:min_c] + idx1[0:min_c]
train_dataset = train_dataset[new_indices,:]
train_labels = train_labels[new_indices]
print(train_dataset.shape)  #sample_size * 16 * 16 *RGB
print(train_labels.shape)  # probability of [0]=>Ground [1]=>Round
    
c0 = 0
c1 = 0
for i in range(len(train_labels)):
    if train_labels[i][0] == 1:
        c0 = c0 + 1
    else:
        c1 = c1 + 1
print ('Number of data points per class: c0 = ' + str(c0) + ' c1 = ' + str(c1))

# GET THE Prediction Data



train_size = train_dataset.shape[0]
print(train_dataset.shape)
print(train_labels.shape)
train_data_node = tf.placeholder(
    tf.float32,
    shape=(BATCH_SIZE, train_dataset.shape[1]))  # X_batch

train_labels_node = tf.placeholder(tf.float32,
                                   shape=(BATCH_SIZE, 2))  # Y_batch

train_all_data_node = tf.constant(train_dataset)

# Define the Network Classifier
fc1_weights = tf.Variable(  # fully connected, depth 512.
    tf.truncated_normal([train_dataset.shape[1], HIDDEN_SIZE],
                        stddev=0.1,
                        seed=SEED))
fc1_biases = tf.Variable(tf.constant(0.1, shape=[HIDDEN_SIZE]))

fc2_weights = tf.Variable(  # fully connected, depth 512.
    tf.truncated_normal([HIDDEN_SIZE, 2],
                        stddev=0.1,
                        seed=SEED))
fc2_biases = tf.Variable(tf.constant(0.1, shape=[2]))

fc1_weights = tf.get_variable('fc1_weights',
                              shape=(train_dataset.shape[1], HIDDEN_SIZE), initializer=tf.contrib.layers.xavier_initializer())
fc2_weights = tf.get_variable('fc2_weights',
                              shape=(HIDDEN_SIZE, 2), initializer=tf.contrib.layers.xavier_initializer())
total_parameters = numpy.sum([numpy.product([xi.value for xi in x.get_shape()]) for x in tf.all_variables()])
print("Total Number Of Parameters :" + str(total_parameters))


def model(data, train=False):
    hidden = tf.nn.sigmoid(tf.matmul(data, fc1_weights) + fc1_biases)
    out = tf.matmul(hidden, fc2_weights) + fc2_biases
    return out


logits = model(train_data_node, True)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=train_labels_node))
regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) +
                tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases))
loss += 5e-4 * regularizers
batch = tf.Variable(0)
learning_rate = tf.train.exponential_decay(
    0.015,  # Base learning rate.
    batch * BATCH_SIZE,  # Current index into the dataset.
    train_dataset.shape[0],  # Decay step.
    0.98,  # Decay rate.
    staircase=True)
optimizer = tf.train.MomentumOptimizer(learning_rate, 0.0).minimize(loss, global_step=batch)
train_prediction = tf.nn.softmax(logits)
saver = tf.train.Saver()
with tf.Session() as s:
    if RESTORE_MODEL:
        # Restore variables from disk.
        #saver.restore(s, FLAGS.train_dir + "/model.ckpt")
        print("Model restored.")
    else:

        tf.initialize_all_variables().run()
        print('Initialized!')
        print('Total number of iterations = ' + str(int(num_epochs * train_size / BATCH_SIZE)))
        training_indices = range(train_dataset.shape[0])
        iteration = 0
        for iepoch in range(EPOCHS):
            print(str(iteration) + " iteration")
            iteration += 1
            perm_indices = numpy.random.permutation(training_indices)
            for step in range(int(train_dataset.shape[0] / BATCH_SIZE)):

                offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
                batch_indices = perm_indices[offset:(offset + BATCH_SIZE)]
                # Compute the offset of the current minibatch in the data.
                # Note that we could use better randomization across epochs.
                batch_data = train_dataset[batch_indices, :]
                batch_labels = train_labels[batch_indices]
                if step % RECORDING_STEP == 0:
                    feed_dict = {train_data_node: batch_data, train_labels_node: batch_labels}

                    _, l, lr, predictions = s.run(
                        [optimizer, loss, learning_rate, train_prediction],
                        feed_dict=feed_dict)

            print('Epoch %.2f' % (float(step) * BATCH_SIZE / train_dataset.shape[0]))
            print('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
            print('Minibatch error: %.1f%%' % error_rate(predictions,
                                                         batch_labels))
            sys.stdout.flush()
        else:

            feed_dict = {train_data_node: batch_data,
                         train_labels_node: batch_labels}

        _, l, lr, predictions = s.run(
            [optimizer, loss, learning_rate, train_prediction],
            feed_dict=feed_dict)
    # Save the variables to disk. Check if the model is better if not, don't store

    #save_path = saver.save(s, FLAGS.train_dir + "/model.ckpt")
    #print("Model saved in file: %s" % save_path)

    final_model = model(train_all_data_node)
    train_all_prediction = tf.nn.softmax(final_model)
    final_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=final_model, labels=train_labels))
    final_loss += 5e-4 * regularizers
    loss, predictions = s.run([final_loss, train_all_prediction])

    print('error on trainning data : %.1f%%' % error_rate(predictions, train_labels))
    print('loss on trainning data: %.3f' % loss)

    # Now Prediction
    p_dir = './pre_result/'
    s_dir = './post_result/'
    for i in range(33,34):
        imageid =  "%.3d"%i
        filename = 'prediction_'+str(imageid)+'.png'
        img = mpimg.imread(p_dir+filename)
        p_img = get_prediction(img)
        img_3c = convertToPNG(p_img)
        Image.fromarray(img_3c).save(s_dir+filename)
    