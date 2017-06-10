""" Seda Doganay S004299 Department of Computer Science """

import read_input
from ConvNN import ConvNN


class seda_doganay_training(object):

    def __init__(self,network_type="network_1",dataset_type="28x28_dataset",MNIST_path="/tmp/data"):
        self._network_type = network_type
        
        self._dataset_type = dataset_type
        self._MNIST_path = MNIST_path

        self._mnist = read_input.read_data_sets(MNIST_path, one_hot=True)

        if(self._network_type == 'network_1'):
            self.run_network_1()

        elif(self._network_type == 'network_2'):
            self.run_network_2()


    @property
    def network_type(self):
        return self._network_type

    @property
    def dataset_type(self):
        return self._dataset_type
    @property
    def MNIST_path(self):
        return self._MNIST_path

    def run_network_1(self):
        hyper_parameters = {
            'learning_rate': 0.01,
            'training_iters': 20000,
            'batch_size': 64,
            'display_step': 32,

            'conv1_feature_maps': 32,
            'conv2_feature_maps': 64,
            'conv1_patch_size': 5,
            'conv1_stride': 1,
            'conv2_patch_size': 5,
            'conv2_stride': 1,
            'maxp1_patch_size': 2,
            'maxp1_stride': 2,
            'maxp2_patch_size': 2,
            'maxp2_stride': 2,

            'maxp2_fc_shape': 7 * 7 * 64,
            'fc_shape': 1024
        }

        model_name = self.network_type + "_" + self.dataset_type

        if (self._dataset_type == '28x28_dataset'):
            ConvNN(dataset=self._mnist, img_shape=28, model_name=model_name, hyper_parameters=hyper_parameters)
        elif (self._dataset_type == '14x14_dataset'):
            ConvNN(dataset=self._mnist, img_shape=14, model_name=model_name, hyper_parameters=hyper_parameters)
        elif (self._dataset_type == '14x14_augmented_dataset'):
            ConvNN(dataset=self._mnist, img_shape=14, model_name=model_name, hyper_parameters=hyper_parameters, augment=True)
        else:
            raise ValueError('Unvalid dataset is chosen.')


    def run_network_2(self):
        hyper_parameters = {
            'learning_rate': 0.01,
            'training_iters': 20000,
            'batch_size': 64,
            'display_step': 32,

            'conv1_feature_maps': 64,
            'conv2_feature_maps': 128,
            'conv1_patch_size': 4,
            'conv1_stride': 2,
            'conv2_patch_size': 3,
            'conv2_stride': 1,
            'maxp1_patch_size': 4,
            'maxp1_stride': 2,
            'maxp2_patch_size': 4,
            'maxp2_stride': 2,

            'maxp2_fc_shape': 7 * 7 * 64,
            'fc_shape': 1024
        }

        model_name = self.network_type + "_" + self.dataset_type

        if (self._dataset_type == '28x28_dataset'):
            ConvNN(dataset=self._mnist, img_shape=28, model_name=model_name, hyper_parameters=hyper_parameters)
        elif (self._dataset_type == '14x14_dataset'):
            ConvNN(dataset=self._mnist, img_shape=14, model_name=model_name, hyper_parameters=hyper_parameters)
        elif (self._dataset_type == '14x14_augmented_dataset'):
            ConvNN(dataset=self._mnist, img_shape=14, model_name=model_name, hyper_parameters=hyper_parameters,
                   augment=True)
        else:
            raise ValueError('Unvalid dataset is chosen.')




#TRIALS:

seda_doganay_training(network_type="network_1",dataset_type="28x28_dataset",MNIST_path="/tmp/data")
print "-"*20
seda_doganay_training(network_type="network_1",dataset_type="14x14_dataset",MNIST_path="/tmp/data")
print "-"*20
#seda_doganay_training(network_type="network_1",dataset_type="14x14_augmented_dataset",MNIST_path="/tmp/data")
#print "-"*20

seda_doganay_training(network_type="network_2",dataset_type="28x28_dataset",MNIST_path="/tmp/data")
print "-"*20
seda_doganay_training(network_type="network_2",dataset_type="14x14_dataset",MNIST_path="/tmp/data")
print "-"*20
#seda_doganay_training(network_type="network_2",dataset_type="14x14_augmented_dataset",MNIST_path="/tmp/data")
