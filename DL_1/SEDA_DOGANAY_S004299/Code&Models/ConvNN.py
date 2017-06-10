""" Seda Doganay S004299 Department of Computer Science """


import tensorflow as tf
import data_augmentation

class ConvNN(object):
    n_classes = 10  # MNIST total classes (0-9 digits)
    dropout = 0.75  # Dropout, probability to keep units

    def __init__(self,dataset,img_shape,model_name, hyper_parameters,augment=False):
        if(augment): raise ValueError("Augmentation is not supported yet.")
        self._augment = augment
        self._dataset = dataset
        self._model_name = model_name


        self._learning_rate = hyper_parameters['learning_rate']
        self._training_iters = hyper_parameters['training_iters']
        self._batch_size = hyper_parameters['batch_size']
        self._display_step = hyper_parameters['display_step']
        self._conv1_feature_maps = hyper_parameters['conv1_feature_maps']
        self._conv2_feature_maps = hyper_parameters['conv2_feature_maps']
        self._conv1_patch_size = hyper_parameters['conv1_patch_size']
        self._conv1_stride = hyper_parameters['conv1_stride']
        self._conv2_patch_size = hyper_parameters['conv2_patch_size']
        self._conv2_stride = hyper_parameters['conv2_stride']
        self._maxp1_patch_size = hyper_parameters['maxp1_patch_size']
        self._maxp1_stride = hyper_parameters['maxp1_stride']
        self._maxp2_patch_size = hyper_parameters['maxp2_patch_size']
        self._maxp2_stride = hyper_parameters['maxp2_stride']
        #if(not augment):
        neurons = 7 if img_shape==28 else 4
        self._maxp2_fc_shape = neurons*neurons*self._conv2_feature_maps

        self._fc_shape = hyper_parameters['fc_shape']



        # Network Parameters
        self._n_input = 28*28 # (e.g. 784 for img shape: 28*28)
        self._img_shape = img_shape

        # tf Graph input
        self._x = tf.placeholder(tf.float32, [None, self._n_input])
        self._y = tf.placeholder(tf.float32, [None, self.n_classes])
        self._keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)
        # Store layers weight & bias
        self._weights = {
            # for e.g. 5x5 conv, 1 input, 32 outputs
            'wc1': tf.Variable(tf.random_normal([self._conv1_patch_size, self._conv1_patch_size, 1, self._conv1_feature_maps])),
            # 5x5 conv, 32 inputs, 64 outputs
            'wc2': tf.Variable(tf.random_normal([self._conv2_patch_size, self._conv2_patch_size, self._conv1_feature_maps, self._conv2_feature_maps])),
            # fully connected, 7*7*64 inputs, 1024 outputs
            'wd1': tf.Variable(tf.random_normal([self._maxp2_fc_shape, self._fc_shape])),
            # 1024 inputs, 10 outputs (class prediction)
            'out': tf.Variable(tf.random_normal([self._fc_shape, self.n_classes]))
        }

        self._biases = {
            'bc1': tf.Variable(tf.random_normal([self._conv1_feature_maps])),
            'bc2': tf.Variable(tf.random_normal([self._conv2_feature_maps])),
            'bd1': tf.Variable(tf.random_normal([self._fc_shape])),
            'out': tf.Variable(tf.random_normal([self.n_classes]))
        }

        self.const_run_model()


    # Create some wrappers for simplicity
    def conv2d(self, x, W, b, strides=1):
        # Conv2D wrapper, with bias and relu activation
        x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)

    def maxpool2d(self, x, k=2, s=2, padding='SAME'):
        # MaxPool2D wrapper
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, s, s, 1],
                              padding=padding)

    # Create model
    def conv_net(self, x, weights, biases, dropout):
        if (self._img_shape == 14):
            x = data_augmentation.downsample_all(self._x)
            print("train images are downsampled.")
            if(self._augment):
                #raise ValueError('Augmentation is not avaliable yet.')
                x = data_augmentation.prep_data_augment2(self._x)


        # Reshape input picture
        x = tf.reshape(x, shape=[-1, self._img_shape, self._img_shape, 1])

        # Convolution Layer with ReLU
        conv1 = self.conv2d(x, weights['wc1'], biases['bc1'])
        print("conv1: ", conv1)

        # Max Pooling (down-sampling)
        maxp1 = self.maxpool2d(conv1, k=self._maxp1_patch_size, s=self._maxp1_stride)
        print("maxp1: ", maxp1)

        # Convolution Layer with ReLU
        conv2 = self.conv2d(maxp1, weights['wc2'], biases['bc2'])
        print("conv2: ", conv2)

        # Max Pooling (down-sampling)
        maxp2 = self.maxpool2d(conv2, k=self._maxp2_patch_size, s=self._maxp2_stride)
        print("maxp2: ", maxp2)

        # Fully connected layer
        # Reshape conv2 output to fit fully connected layer input
        fc1 = tf.reshape(maxp2, [-1, weights['wd1'].get_shape().as_list()[0]])
        fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
        fc1 = tf.nn.relu(fc1)
        # Apply Dropout
        fc1 = tf.nn.dropout(fc1, dropout)
        print("fc1: ", fc1)

        # Output, class prediction
        out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
        print("out: ", out)

        return out

    def saveModel(self,session,model_name):
        saver = tf.train.Saver()
        saver.save(session, model_name)
        print("Model is saved as "+model_name)


    def const_run_model(self):

        # Construct model
        pred = self.conv_net(self._x, self._weights, self._biases, self._keep_prob)

        # Define loss and optimizer
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=self._y))
        optimizer = tf.train.AdamOptimizer(learning_rate=self._learning_rate).minimize(cost)

        # Evaluate model
        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(self._y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        # Initializing the variables
        init = tf.initialize_all_variables()

        # Launch the graph
        with tf.Session() as sess:
            sess.run(init)
            step = 1
            # Keep training until reach max iterations
            while step * self._batch_size < self._training_iters:
                batch_x, batch_y = self._dataset.train.next_batch(self._batch_size)
                # Run optimization op (backprop)
                sess.run(optimizer, feed_dict={self._x: batch_x, self._y: batch_y,
                                               self._keep_prob: self.dropout})
                if step % self._display_step == 0:
                    # Calculate batch loss and accuracy
                    loss, acc = sess.run([cost, accuracy], feed_dict={self._x: batch_x,
                                                                      self._y: batch_y,
                                                                      self._keep_prob: 1.})
                    print("Iter " + str(step * self._batch_size) + ", Minibatch Loss= " + \
                          "{:.6f}".format(loss) + ", Training Accuracy= " + \
                          "{:.5f}".format(acc))
                step += 1
            print("Optimization Finished!")

            # Calculate accuracy for mnist test images
            print("Testing Accuracy:",
                  sess.run(accuracy, feed_dict={self._x: self._dataset.test.images,
                                                self._y: self._dataset.test.labels,
                                                self._keep_prob: 1.}))

            self.saveModel(sess, self._model_name)
