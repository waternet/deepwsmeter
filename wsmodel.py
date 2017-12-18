import tensorflow as tf
import glob
from skimage import io
from skimage.transform import resize
from sklearn.model_selection import train_test_split
import numpy as np

# initialize variables to hold all images and labels
X_data = []
y_data = []

#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


#print(mnist.train.next_batch(100)[1].shape)

# load all data
def load_data():
    files = [filename for filename in glob.iglob('data/getallen/test/*/*.jpg', recursive=True)]
    for file in files:
        img = io.imread(file, as_grey=True)
        img = resize(img, (28, 28))
        X_data.append(np.ndarray.flatten(img))
        y_data.append(int(file.split('/')[3]))

load_data()

# convert to a numpy array
X_data = np.array(X_data)

# split into test train_data
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.33, random_state=28)

# one hot encode the labels
y_train = tf.one_hot(y_train, depth=10)
y_test = tf.one_hot(y_test, depth=10)

print(X_train.shape)
print(y_train.shape)
# build the model
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def get_batch(i, batchsize):
    X = X_train[(i-1)*batchsize:i*batchsize]
    y = y_train[(i-1)*batchsize:i*batchsize]
    return (X, y)

batchsize = 100
for i in range(int(X_train.shape[0] / batchsize) + 1):
    get_batch(i+1, batchsize)

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

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(2000):
        #batch = mnist.train.next_batch(100)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: X_train, y_: y_train.eval(), keep_prob: 1.0})
            print('step %d, training accuracy %g' % (i, train_accuracy))
        train_step.run(feed_dict={x: X_train, y_: y_train.eval(), keep_prob: 0.5})

    print('test accuracy %g' % accuracy.eval(feed_dict={
        x: X_test, y_: y_test.eval(), keep_prob: 1.0}))

    save_path = saver.save(sess, "models/wsmodel.ckpt")
    print("Model saved in file: ", save_path)
