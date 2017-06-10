#THREADING NOT AVAILABLE!
import cifar10_input

batch_size = 128
data_path = '/tmp/cifar10_data'
train_dir = '/tmp/cifar10_train'
eval_dir = '/tmp/cifar10_eval'
NUM_CLASSES = cifar10_input.NUM_CLASSES

conv1_w_shape = [5, 5, 3, 64]
conv1_b_shape = [conv1_w_shape[3]]
conv1_b = 0.0

#norm1

pool1_ksize_shape = [1, 3, 3, 1]
pool1_stride_shape = [1, 2, 2, 1]

conv2_w_shape = [5, 5, 64, 64]
conv2_b_shape = [conv2_w_shape[3]]
conv2_b = 0.1

#norm2

pool2_ksize_shape = [1, 3, 3, 1]
pool2_stride_shape = [1, 2, 2, 1]

local3_w_dim = 384
local3_w_shape = [] #not directly in my control
local3_b_shape = [local3_w_dim]
local3_b = 0.1

local4_w_shape = [local3_w_dim, 192]
local4_b_shape = [local4_w_shape[1]]
local4_b = 0.1

softmax_w_shape = [192, NUM_CLASSES]
softmax_b_shape = [NUM_CLASSES]


def calc_hiddens():
    hidden = 0
    hidden += mul_arr_elements(conv1_w_shape)
    hidden += mul_arr_elements(conv1_b_shape)

    hidden += mul_arr_elements(conv2_w_shape)
    hidden += mul_arr_elements(conv2_b_shape)

    hidden += mul_arr_elements(local3_w_shape)
    hidden += mul_arr_elements(local3_b_shape)

    hidden += mul_arr_elements(local4_w_shape)
    hidden += mul_arr_elements(local4_b_shape)

    hidden += mul_arr_elements(softmax_w_shape)
    hidden += mul_arr_elements(softmax_b_shape)

    return hidden

def mul_arr_elements(arr):
    temp = 1
    for i in arr:
        temp = temp * i
    return temp


#YEDEK is below:
# conv1_w_shape = [5, 5, 3, 64]
# conv1_b_shape = [conv1_w_shape[3]]
# conv1_b = 0.0
#
# #norm1
#
# pool1_ksize_shape = [1, 3, 3, 1]
# pool1_stride_shape = [1, 2, 2, 1]
#
# conv2_w_shape = [5, 5, 64, 64]
# conv2_b_shape = [conv2_w_shape[3]]
# conv2_b = 0.1
#
# #norm2
#
# pool2_ksize_shape = [1, 3, 3, 1]
# pool2_stride_shape = [1, 2, 2, 1]
#
# local3_w_dim = 384
# local3_w_shape = [] #not directly in my control
# local3_b_shape = [local3_w_dim]
# local3_b = 0.1
#
# local4_w_shape = [local3_w_dim, 192]
# local4_b_shape = [local4_w_shape[1]]
# local4_b = 0.1
#
# softmax_w_shape = [192, NUM_CLASSES]
# softmax_b_shape = [NUM_CLASSES]