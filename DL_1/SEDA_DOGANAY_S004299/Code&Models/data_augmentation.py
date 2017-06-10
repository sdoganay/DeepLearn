""" Seda Doganay S004299 Department of Computer Science """


import tensorflow as tf


learning_rate = 0.001
training_iters = 2000
batch_size = 16
display_step = 10


# Network Parameters
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)


def maxpool2d(x, k=2,s=2,padding='SAME'):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, s, s, 1],
                          padding=padding)


# Create model
def downsample_size(x):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # Max Pooling (down-sampling)
    maxp1 = maxpool2d(x, k=2,s=2,padding='VALID')

    return maxp1


def prep_data_augment1(image):
    # Reshape input picture
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=63/255.0)
    return image

def prep_data_augment2(images):
    from random import randint
    image = tf.image.rotate(images, randint(-45,45))
    image = tf.image.random_brightness(image, max_delta=63/255.0)
    return image

def prep_data_augment3(image):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=63/255.0)
    return image


def downsample_all(input_tensor):

    output_tensor = tf.map_fn(downsample_size, input_tensor)

    print("out shape: ", output_tensor)

    return output_tensor

def augment_all(input_tensor):
    output_tensor = tf.map_fn(prep_data_augment1, input_tensor)
    print("aug shape: ", output_tensor)
    return output_tensor

# TRIAL:
#with tf.Session() as sess:
#    out = downsample_all(mnist.train.images)
#    result = sess.run([out])