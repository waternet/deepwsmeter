import glob
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.transform import resize
import tensorflow as tf
from skimage import io
import scipy

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

save_file = './models/wsmodel.ckpt'


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

def predict(image):
    tf.reset_default_graph()

    img = io.imread(image, as_grey=True)
    img = resize(img, (28, 28))
    #img = scipy.misc.imfilter(img, 'blur')

    X_new = np.ndarray.flatten(img)

    #plt.imshow(np.resize(X_new, (28, 28)), cmap=plt.cm.gray)
    #plt.show()

    x = tf.placeholder(tf.float32, shape=[None, 784])

    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')

    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])

    x_image = tf.reshape(x, [-1, 28, 28, 1])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        saver.restore(sess, save_file)
        prediction = tf.argmax(y_conv, 1)
        return prediction.eval(feed_dict={x: [X_new], keep_prob: 1.0}, session = sess)

if __name__ == "__main__":
    import random
    import matplotlib.image as mpimg
    all_testfiles = [filename for filename in glob.iglob('data/getallen/test/*/*.jpg', recursive=True)]
    random.shuffle(all_testfiles)

    images = []
    labels = []
    for i in range(0, 10):
        images.append(mpimg.imread(all_testfiles[i]))
        labels.append(predict(all_testfiles[i]))
