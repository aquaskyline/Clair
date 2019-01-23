import tensorflow as tf
from tensorflow.python.client import device_lib
import numpy as np
import sys
import argparse
import io
import os
import re
import selu
import param

from collections import defaultdict
import multiprocessing
import imageio
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1 import make_axes_locatable

class Qianliyan(object):
    """
    Modification of Clairvoyante v3

    Keywords arguments:

    float_type: The type of float to be used for tensorflow, default tf.float64
    input_shape: Shpae of the input tensor, a tuple or list of 3 integers

    task_loss_weights: The weights of different tasks in the calculation of total loss, list of 5 integers in the order (base_change, zygosity, variant type, indel length, L2 regularization)
    structure: The name of the structure, supporting "FC_L3_narrow_legacy_0.1, 2BiLST, CNN1D_L6, Res1D_L9M"

    output_base_change_shape: The number of classes in the ouput of base change (alternate base) prediction
    output_zygosity_shape: The number of classes in the ouput of zygosity prediction
    output_variant_type_shape: The number of classes in the ouput of variant type (No variant, SNP, insertion, deletion, etc.) prediction
    output_indel_length_shape: The number of classes in the ouput of indel length prediction

    output_weight_enabled: True enables per class weights speficied in output_*_entropy_weights (Slower)
    output_base_change_entropy_weights: A list of (output_base_change_shape) integers specifying the weights of different classes in the calculation of entropy loss
                    (Only used when output_weight_enabled is set to True)
    output_zygosity_entropy_weights, output_variant_type_entropy_weights, output_indel_length_entropy_weights: similar to output_base_change_entropy_weights
    
    L1_num_units: Number of units in L1

    tensor_transform_function: the function (callable) for transforming the input tensors to match the model, takes in X_tensor, Y_tensor, and stage text ("train" or "predict") and
            return the pair (transformed_X, transformed_Y)
            i.e. type: tensor -> tensor -> str -> (tensor, tensor)
            default: lambda X, Y, phase: (X, Y)  (identity ignoring stage text), which is equivalent to def f(X, Y, phase): return (X, Y)
            
    """
    COLORS_RGB = dict(
        RED = [1.0, 0.0, 0.0],
        GREEN = [0.0, 1.0, 0.0],
        BLUE = [0.0, 0.0, 1.0],
        WHITE = [1.0, 1.0, 1.0],
        BLACK = [0.0, 0.0, 0.0]
    )

    def __init__(self, **kwargs):
        params = dict(
            float_type = tf.float64,
            input_shape = (2 * param.flankingBaseNum + 1, 4, param.matrixNum),  # (17, 4, 4)
            task_loss_weights = [1, 1, 1, 1, 1],
            structure = "2BiLSTM",
            output_base_change_shape = 4,
            output_zygosity_shape = 2, 
            output_variant_type_shape = 4,
            output_indel_length_shape = 6,
            output_base_change_entropy_weights = [1, 1, 1, 1],
            output_zygosity_entropy_weights = [1, 1],
            output_variant_type_entropy_weights = [1, 1, 1, 1],
            output_indel_length_entropy_weights = [1, 1, 1, 1, 1, 1],
            output_weight_enabled = False,
            L1_num_units = 30,
            L2_num_units = 30,
            L3_num_units = 192,
            L3_dropout_rate = 0.5,
            LSTM1_num_units = 256,
            LSTM2_num_units = 256,
            LSTM1_dropout_rate = 0.0,
            LSTM2_dropout_rate = 0.5,
            initial_learning_rate = param.initialLearningRate,
            learning_rate_decay = param.learningRateDecay,
            l2_regularization_lambda = param.l2RegularizationLambda, #  0.001
            l2_regularization_lambda_decay_rate = param.l2RegularizationLambdaDecay,
            tensor_transform_function = lambda X, Y, phase: (X, Y)
        )

        print("This is Qianliyan")

        # Getting parameters from the param.py file
        params_from_file = param.get_model_parameters()
        params.update(params_from_file)
        for key, value in kwargs.items():
            if key in params.keys():
                params[key] = value
            else:
                print("Info: the parameter %s, with value %s is not supported" % (key, value))
        print(params)

        self.input_shape = params['input_shape']
        self.tensor_transform_function = params['tensor_transform_function']
        self.output_base_change_shape = params['output_base_change_shape']
        self.output_zygosity_shape = params['output_zygosity_shape']
        self.output_variant_type_shape = params['output_variant_type_shape']
        self.output_indel_length_shape = params['output_indel_length_shape']

        self.task_loss_weights = np.array(params['task_loss_weights'], dtype = float)

        self.output_weight_enabled = params['output_weight_enabled']
        self.output_base_change_entropy_weights = np.array(params['output_base_change_entropy_weights'], dtype = float)
        self.output_zygosity_entropy_weights = np.array(params['output_zygosity_entropy_weights'], dtype = float)
        self.output_variant_type_entropy_weights = np.array(params['output_variant_type_entropy_weights'], dtype = float)
        self.output_indel_length_entropy_weights = np.array(params['output_indel_length_entropy_weights'], dtype = float)

        self.L1_num_units = params['L1_num_units']
        self.L2_num_units = params['L2_num_units']
        self.L3_num_units = params['L3_num_units']
        self.L3_dropout_rate = params['L3_dropout_rate']

        self.LSTM1_num_units = params['LSTM1_num_units']
        self.LSTM2_num_units = params['LSTM2_num_units']
        self.LSTM1_dropout_rate = params['LSTM1_dropout_rate']
        self.LSTM2_dropout_rate = params['LSTM2_dropout_rate']

        self.learning_rate_value = params['initial_learning_rate']
        self.learning_rate_decay_rate = params['learning_rate_decay']
        self.l2_regularization_lambda_value = params['l2_regularization_lambda']
        self.l2_regularization_lambda_decay_rate = params['l2_regularization_lambda_decay_rate']
        self.structure = params['structure']
        
        if 'CNN' in self.structure or 'Res' in self.structure or 'LSTM' in self.structure:
            self.float_type = tf.float32
        else:
            self.float_type = params['float_type']

        self.output_cache = {}
        self.output_label_split = [self.output_base_change_shape, self.output_zygosity_shape, self.output_variant_type_shape, self.output_indel_length_shape]

        print(self.input_shape)
        
        self.g = tf.Graph()
        self._build_graph()
        self.session = tf.Session(graph = self.g, config=tf.ConfigProto(intra_op_parallelism_threads=param.NUM_THREADS))

    @staticmethod
    def get_available_gpus():
        local_device_protos = device_lib.list_local_devices()
        return [x.name for x in local_device_protos if x.device_type == 'GPU']

    def get_structure_dict(self, phase = 'train'):
        
        if phase == 'train':
            return {self.L3_dropout_rate_placeholder:self.L3_dropout_rate}
        else:
            return {self.L3_dropout_rate_placeholder:0.0}

    @staticmethod
    def slice_dense_layer(inputs, units, slice_dimension, name="slice_dense", **kwargs):
        """
        Specify a slice dense layer, which unpacks along the specified dimension and connects each position to another layer by full connections
        e.g. A tensor of shape [4, 5] would be unpacked to 4 tensors with shape [5], and each of the tensor with shape [5] is fully connected
             to another tensor with [units], and restacked to give a tensor with output shape [4, units]

        inputs: The input tensor
        units: The number of units for each position
        slice_dimension: The index of the dimension to be sliced, following the order of tensor.shape
        name: The name of the operation (variable scope)
        **kwargs: Other parameters to be passed to the tf.layers.dense() function
        """
        with tf.variable_scope(name):
            sliced = tf.unstack(inputs, axis=slice_dimension, name=name + "Unstack")
            slice_dense = tf.stack([tf.layers.dense(v, units=units, name="Unit_" + str(i), **kwargs) for i, v in enumerate(sliced)], axis=slice_dimension, name="Stacked")
            return slice_dense

    @staticmethod
    def weighted_cross_entropy(softmax_prediction, labels, weights, epsilon, **kwargs):
        """
        Compute cross entropy with per class weights
        """
        return -tf.reduce_sum(tf.multiply(labels * tf.log(softmax_prediction + epsilon), weights), reduction_indices=[1], **kwargs)

    @staticmethod
    def compatible_BiLSTM_layer(inputs, num_units, name="comptatible_BiLSTM", cudnn_gpu_available=False):
        with tf.variable_scope(name):
            if cudnn_gpu_available:
                # inputs = tf.random_uniform(shape, dtype=tf.float32)
                lstm = tf.contrib.cudnn_rnn.CudnnLSTM(
                    num_layers=1,
                    num_units=num_units,
                    direction='bidirectional',
                    dtype=tf.float32)
                lstm.build(inputs.get_shape())
                outputs, output_states = lstm(inputs, training=True)
                print(outputs, output_states)
                return outputs, output_states
            else:
                single_cell_generator = lambda: tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(num_units) #, reuse=tf.get_variable_scope().reuse
                # inputs = tf.random_uniform(shape, dtype=tf.float32)
                lstm_fw_cells = [single_cell_generator()]
                lstm_bw_cells = [single_cell_generator()]
                (outputs, output_state_fw, output_state_bw) = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                    lstm_fw_cells,
                    lstm_bw_cells,
                    inputs,
                    dtype=tf.float32,
                    time_major=True)
                print(outputs, (output_state_fw, output_state_bw))
                return outputs, (output_state_fw, output_state_bw)

    def _build_graph(self):
        # self.graph = tf.Graph()
        self.graph = self.g
        self.layers = []
        with self.graph.as_default():
            # Conversion
            self.epsilon = tf.constant(value=1e-10, dtype=self.float_type)
            self.input_shape_tf = (None, self.input_shape[0], self.input_shape[1], self.input_shape[2])
            
            # Place holders
            self.X_placeholder = tf.placeholder(self.float_type, self.input_shape_tf, name='X_placeholder')
            self.Y_placeholder = tf.placeholder(self.float_type, [None, 
                        self.output_base_change_shape + self.output_zygosity_shape + self.output_variant_type_shape + self.output_indel_length_shape], name='Y_placeholder')
            self.layers.append(self.X_placeholder)

            self.learning_rate_placeholder = tf.placeholder(self.float_type, shape=[], name='learning_rate_placeholder')
            self.phase_placeholder = tf.placeholder(tf.bool, shape=[], name='phase_placeholder')
            self.regularization_L2_lambda_placeholder = tf.placeholder(self.float_type, shape=[], name='regularization_L2_lambda_placeholder')
            self.task_loss_weights_placeholder = tf.placeholder(self.float_type, shape=self.task_loss_weights.shape, name='task_loss_weights_placeholder')
            self.output_base_change_entropy_weights_placeholder = tf.placeholder(self.float_type, shape=self.output_base_change_entropy_weights.shape, name='output_base_change_entropy_weights_placeholder')
            self.output_zygosity_entropy_weights_placeholder = tf.placeholder(self.float_type, shape=self.output_zygosity_entropy_weights.shape, name='output_zygosity_entropy_weights_placeholder')
            self.output_variant_type_entropy_weights_placeholder = tf.placeholder(self.float_type, shape=self.output_variant_type_entropy_weights.shape, name='output_variant_type_entropy_weights_placeholder')
            self.output_indel_length_entropy_weights_placeholder = tf.placeholder(self.float_type, shape=self.output_indel_length_entropy_weights.shape, name='output_indel_length_entropy_weights_placeholder')

            if self.structure == "FC_L3_narrow" or self.structure == "FC_L3_narrow_legacy_0.1":
            
                self.L3_dropout_rate_placeholder = tf.placeholder(self.float_type, shape=[], name='L3_dropout_rate_placeholder')

                # Flatten the 2nd (ACGT) and 3rd (Ref Ins Del SNP) dimension
                self.X_flattened_2D = tf.reshape(self.X_placeholder, shape=(tf.shape(self.X_placeholder)[0], self.input_shape_tf[1], self.input_shape_tf[2] * self.input_shape_tf[3]), 
                            name="X_flattened_2D")
                print(self.X_flattened_2D.shape)
                self.layers.append(self.X_flattened_2D)
                
                # Slice dense layer 1
                self.L1 = Qianliyan.slice_dense_layer(self.X_flattened_2D, self.L1_num_units, slice_dimension = 1, name="L1", activation=selu.selu)
                self.layers.append(self.L1)

                # Slice dense layer 2
                self.L2 = Qianliyan.slice_dense_layer(self.L1, self.L2_num_units, slice_dimension = 2, name="L2", activation=selu.selu)
                self.layers.append(self.L2)

                self.L2_flattened = tf.reshape(self.L2, shape=(tf.shape(self.L2)[0], self.L2_num_units * self.L1_num_units), name="L2_flattened")

                # Dense layer 3
                self.L3 = tf.layers.dense(self.L2_flattened, units=self.L3_num_units, name="L3", activation=selu.selu)
                self.layers.append(self.L3)

                self.L3_dropout = selu.dropout_selu(self.L3, self.L3_dropout_rate_placeholder, training=self.phase_placeholder, name='L3_dropout')
                self.layers.append(self.L3_dropout)

                self.core_final_layer = self.L3_dropout
            
            if self.structure == "CNN1D_L6":
                self.float_type = tf.float32

                self.L3_dropout_rate_placeholder = tf.placeholder(self.float_type, shape=[], name='L3_dropout_rate_placeholder')

                # Flatten the 2nd (ACGT) and 3rd (Ref Ins Del SNP) dimension
                self.X_flattened_2D = tf.reshape(self.X_placeholder, shape=(tf.shape(self.X_placeholder)[0], self.input_shape_tf[1], self.input_shape_tf[2] * self.input_shape_tf[3]), 
                            name="X_flattened_2D")

                # Slice dense layer 1
                self.L1 = Qianliyan.slice_dense_layer(self.X_flattened_2D, self.L1_num_units, slice_dimension = 1, name="L1", activation=selu.selu)

                with tf.variable_scope("Res1"):
                    self.Res1_Conv1 = tf.layers.conv1d(self.L1, filters=128, kernel_size=3, strides=1, padding='same', name="Conv1", activation=selu.selu, kernel_initializer=tf.keras.initializers.he_normal())
                    self.Res1_dropout1 = selu.dropout_selu(self.Res1_Conv1, 0.3, training=self.phase_placeholder, name='Conv1_dropout')
                    self.Res1_Conv2 = tf.layers.conv1d(self.Res1_dropout1, filters=self.L1_num_units, kernel_size=3, strides=1, padding='same', name="Conv2", activation=selu.selu, kernel_initializer=tf.keras.initializers.he_normal())
                    self.Res1_sum = tf.add(self.L1, self.Res1_Conv2, name="sum")
                
                with tf.variable_scope("Res2"):
                    self.Res2_Conv1 = tf.layers.conv1d(self.Res1_sum, filters=128, kernel_size=3, strides=1, padding='same', name="Conv1", activation=selu.selu, kernel_initializer=tf.keras.initializers.he_normal())
                    self.Res2_dropout1 = selu.dropout_selu(self.Res2_Conv1, 0.3, training=self.phase_placeholder, name='Conv1_dropout')
                    self.Res2_Conv2 = tf.layers.conv1d(self.Res2_dropout1, filters=self.L1_num_units, kernel_size=3, strides=1, padding='same', name="Conv2", activation=selu.selu, kernel_initializer=tf.keras.initializers.he_normal())
                    self.Res2_sum = tf.add(self.Res1_sum, self.Res2_Conv2, name="sum")
                
                with tf.variable_scope("Res3"):
                    self.Res3_Conv1 = tf.layers.conv1d(self.Res2_sum, filters=128, kernel_size=3, strides=1, padding='same', name="Conv1", activation=selu.selu, kernel_initializer=tf.keras.initializers.he_normal())
                    self.Res3_dropout1 = selu.dropout_selu(self.Res3_Conv1, 0.3, training=self.phase_placeholder, name='Conv1_dropout')
                    self.Res3_Conv2 = tf.layers.conv1d(self.Res3_dropout1, filters=self.L1_num_units, kernel_size=3, strides=1, padding='same', name="Conv2", activation=selu.selu, kernel_initializer=tf.keras.initializers.he_normal())
                    self.Res3_sum = tf.add(self.Res2_sum, self.Res3_Conv2, name="sum")
                
                # Slice dense layer 2
                self.L2 = Qianliyan.slice_dense_layer(self.Res3_sum, self.L2_num_units, slice_dimension = 2, name="L2", activation=selu.selu)

                self.L2_flattened = tf.reshape(self.L2, shape=(tf.shape(self.L2)[0], self.L2_num_units * self.L1_num_units), name="L2_flattened")

                self.L2_dropout = selu.dropout_selu(self.L2_flattened, 0.5, training=self.phase_placeholder, name='L2_dropout')

                # Dense layer 3
                self.L3 = tf.layers.dense(self.L2_dropout, units=self.L3_num_units, name="L3", activation=selu.selu)

                self.L3_dropout = selu.dropout_selu(self.L3, self.L3_dropout_rate_placeholder, training=self.phase_placeholder, name='L3_dropout')

                # Dense layer 4
                self.L4 = tf.layers.dense(self.L3_dropout, units=self.L3_num_units, name="L4", activation=selu.selu)

                # self.L4_dropout = selu.dropout_selu(self.L4, self.L3_dropout_rate_placeholder, training=self.phase_placeholder, name='L4_dropout')

                self.core_final_layer = self.L4

            if self.structure == "Res1D_L9":
                self.float_type = tf.float32

                self.L3_dropout_rate_placeholder = tf.placeholder(self.float_type, shape=[], name='L3_dropout_rate_placeholder')

                # Flatten the 2nd (ACGT) and 3rd (Ref Ins Del SNP) dimension
                self.X_flattened_2D = tf.reshape(self.X_placeholder, shape=(tf.shape(self.X_placeholder)[0], self.input_shape_tf[1], self.input_shape_tf[2] * self.input_shape_tf[3]), 
                            name="X_flattened_2D")

                self.conv1_num_units = 50
                self.conv1 = tf.layers.conv1d(self.X_flattened_2D, filters=self.conv1_num_units, kernel_size=3, strides=1, padding='same', name="Conv1", activation=selu.selu, kernel_initializer=tf.keras.initializers.he_normal())

                with tf.variable_scope("Res1"):
                    self.Res1_Conv1 = tf.layers.conv1d(self.conv1, filters=128, kernel_size=3, strides=1, padding='same', name="Conv1", activation=selu.selu, kernel_initializer=tf.keras.initializers.he_normal())
                    self.Res1_dropout1 = selu.dropout_selu(self.Res1_Conv1, 0.3, training=self.phase_placeholder, name='Conv1_dropout')
                    self.Res1_Conv2 = tf.layers.conv1d(self.Res1_dropout1, filters=self.conv1_num_units, kernel_size=3, strides=1, padding='same', name="Conv2", activation=selu.selu, kernel_initializer=tf.keras.initializers.he_normal())
                    self.Res1_sum = tf.add(self.conv1, self.Res1_Conv2, name="sum")
                
                with tf.variable_scope("Res2"):
                    self.Res2_Conv1 = tf.layers.conv1d(self.Res1_sum, filters=128, kernel_size=3, strides=1, padding='same', name="Conv1", activation=selu.selu, kernel_initializer=tf.keras.initializers.he_normal())
                    self.Res2_dropout1 = selu.dropout_selu(self.Res2_Conv1, 0.3, training=self.phase_placeholder, name='Conv1_dropout')
                    self.Res2_Conv2 = tf.layers.conv1d(self.Res2_dropout1, filters=self.conv1_num_units, kernel_size=3, strides=1, padding='same', name="Conv2", activation=selu.selu, kernel_initializer=tf.keras.initializers.he_normal())
                    self.Res2_sum = tf.add(self.Res1_sum, self.Res2_Conv2, name="sum")
                
                with tf.variable_scope("Res3"):
                    self.Res3_Conv1 = tf.layers.conv1d(self.Res2_sum, filters=128, kernel_size=3, strides=1, padding='same', name="Conv1", activation=selu.selu, kernel_initializer=tf.keras.initializers.he_normal())
                    self.Res3_dropout1 = selu.dropout_selu(self.Res3_Conv1, 0.3, training=self.phase_placeholder, name='Conv1_dropout')
                    self.Res3_Conv2 = tf.layers.conv1d(self.Res3_dropout1, filters=self.conv1_num_units, kernel_size=3, strides=1, padding='same', name="Conv2", activation=selu.selu, kernel_initializer=tf.keras.initializers.he_normal())
                    self.Res3_sum = tf.add(self.Res2_sum, self.Res3_Conv2, name="sum")

                with tf.variable_scope("Res4"):
                    self.Res4_Conv1 = tf.layers.conv1d(self.Res3_sum, filters=128, kernel_size=3, strides=1, padding='same', name="Conv1", activation=selu.selu, kernel_initializer=tf.keras.initializers.he_normal())
                    self.Res4_dropout1 = selu.dropout_selu(self.Res4_Conv1, 0.3, training=self.phase_placeholder, name='Conv1_dropout')
                    self.Res4_Conv2 = tf.layers.conv1d(self.Res4_dropout1, filters=self.conv1_num_units, kernel_size=3, strides=1, padding='same', name="Conv2", activation=selu.selu, kernel_initializer=tf.keras.initializers.he_normal())
                    self.Res4_sum = tf.add(self.Res3_sum, self.Res4_Conv2, name="sum")
                
                with tf.variable_scope("Res5"):
                    self.Res5_Conv1 = tf.layers.conv1d(self.Res4_sum, filters=128, kernel_size=3, strides=1, padding='same', name="Conv1", activation=selu.selu, kernel_initializer=tf.keras.initializers.he_normal())
                    self.Res5_dropout1 = selu.dropout_selu(self.Res5_Conv1, 0.3, training=self.phase_placeholder, name='Conv1_dropout')
                    self.Res5_Conv2 = tf.layers.conv1d(self.Res5_dropout1, filters=self.conv1_num_units, kernel_size=3, strides=1, padding='same', name="Conv2", activation=selu.selu, kernel_initializer=tf.keras.initializers.he_normal())
                    self.Res5_sum = tf.add(self.Res4_sum, self.Res5_Conv2, name="sum")
                
                with tf.variable_scope("Res6"):
                    self.Res6_Conv1 = tf.layers.conv1d(self.Res5_sum, filters=128, kernel_size=3, strides=1, padding='same', name="Conv1", activation=selu.selu, kernel_initializer=tf.keras.initializers.he_normal())
                    self.Res6_dropout1 = selu.dropout_selu(self.Res6_Conv1, 0.3, training=self.phase_placeholder, name='Conv1_dropout')
                    self.Res6_Conv2 = tf.layers.conv1d(self.Res6_dropout1, filters=self.conv1_num_units, kernel_size=3, strides=1, padding='same', name="Conv2", activation=selu.selu, kernel_initializer=tf.keras.initializers.he_normal())
                    self.Res6_sum = tf.add(self.Res5_sum, self.Res6_Conv2, name="sum")

                # # Slice dense layer 1
                # self.L1 = Qianliyan.slice_dense_layer(self.X_flattened_2D, self.L1_num_units, slice_dimension = 1, name="L1", activation=selu.selu)
                
                # Slice dense layer 2
                self.L2 = Qianliyan.slice_dense_layer(self.Res6_sum, self.L2_num_units, slice_dimension = 2, name="L2", activation=selu.selu)

                self.L2_flattened = tf.reshape(self.L2, shape=(tf.shape(self.L2)[0], self.L2_num_units * self.conv1_num_units), name="L2_flattened")

                self.L2_dropout = selu.dropout_selu(self.L2_flattened, 0.5, training=self.phase_placeholder, name='L2_dropout')

                # Dense layer 3
                self.L3 = tf.layers.dense(self.L2_dropout, units=self.L3_num_units, name="L3", activation=selu.selu)

                self.L3_dropout = selu.dropout_selu(self.L3, self.L3_dropout_rate_placeholder, training=self.phase_placeholder, name='L3_dropout')

                # Dense layer 4
                self.L4 = tf.layers.dense(self.L3_dropout, units=self.L3_num_units, name="L4", activation=selu.selu)

                # self.L4_dropout = selu.dropout_selu(self.L4, self.L3_dropout_rate_placeholder, training=self.phase_placeholder, name='L4_dropout')

                self.core_final_layer = self.L4

            if self.structure == "2BiLSTM":
                self.L3_dropout_rate_placeholder = tf.placeholder(self.float_type, shape=[], name='L3_dropout_rate_placeholder')

                # Flatten the 2nd (ACGT) and 3rd (Ref Ins Del SNP) dimension
                self.X_flattened_2D = tf.reshape(self.X_placeholder, shape=(tf.shape(self.X_placeholder)[0], self.input_shape_tf[1], self.input_shape_tf[2] * self.input_shape_tf[3]), 
                            name="X_flattened_2D")
                self.layers.append(self.X_flattened_2D)

                self.X_flattened_2D_transposed = tf.transpose(self.X_flattened_2D, [1, 0, 2], name="X_flattened_2D_transposed")

                is_gpu_available = len(Qianliyan.get_available_gpus()) > 0
                print("is_gpu_available:", is_gpu_available)
                self.LSTM1, _ = Qianliyan.compatible_BiLSTM_layer(self.X_flattened_2D_transposed, self.LSTM1_num_units, name="LSTM1", cudnn_gpu_available=is_gpu_available)
                self.layers.append(self.LSTM1)
                self.LSTM1_dropout = tf.layers.dropout(self.LSTM1,rate=self.LSTM1_dropout_rate, training=self.phase_placeholder,name="LSTM1_dropout")
                self.LSTM2, _ = Qianliyan.compatible_BiLSTM_layer(self.LSTM1_dropout, self.LSTM2_num_units, name="LSTM2", cudnn_gpu_available=is_gpu_available)
                self.layers.append(self.LSTM2)
                self.LSTM2_dropout = tf.layers.dropout(self.LSTM2, rate=self.LSTM2_dropout_rate, training=self.phase_placeholder, name="LSTM2_dropout")
                self.LSTM2_transposed = tf.transpose(self.LSTM2_dropout, [1, 0, 2], name="LSTM2_transposed")

                # self.LSTM1_object = tf.contrib.cudnn_rnn.CudnnLSTM(
                #     num_layers=1,
                #     num_units=self.LSTM1_num_units,
                #     direction="bidirectional",
                #     name="LSTM1")
                # self.LSTM1, _ = self.LSTM1_object(self.X_flattened_2D_transposed)
                # self.layers.append(self.LSTM1)
                # self.LSTM1_dropout = tf.layers.dropout(self.LSTM1,rate=self.LSTM1_dropout_rate, training=self.phase_placeholder,name="LSTM1_dropout")
                # self.LSTM2_object = tf.contrib.cudnn_rnn.CudnnLSTM(
                #     num_layers=1,
                #     num_units=self.LSTM2_num_units,
                #     direction="bidirectional",
                #     name="LSTM2")
                # self.LSTM2, _ = self.LSTM2_object(self.LSTM1_dropout)
                # self.layers.append(self.LSTM2)
                # self.LSTM2_dropout = tf.layers.dropout(self.LSTM2, rate=self.LSTM2_dropout_rate, training=self.phase_placeholder, name="LSTM2_dropout")
                # # Convert back from time-major outputs to batch-major outputs.
                # self.LSTM2_transposed = tf.transpose(self.LSTM2_dropout, [1, 0, 2], name="LSTM2_transposed")
                # # self.LSTM2_transposed_reshaped = tf.reshape(self.LSTM2_transposed, shape=(tf.shape(self.X_placeholder)[0], self.input_shape_tf[1], 8 * 2), name="LSTM2_transposed_reshaped")


                # Slice dense layer 1
                # self.L1 = Qianliyan.slice_dense_layer(self.X_flattened_2D, self.L1_num_units, slice_dimension = 1, name="L1", activation=selu.selu)
                # self.layers.append(self.L1)

                # Slice dense layer 2
                self.L2 = Qianliyan.slice_dense_layer(self.LSTM2_transposed, self.L2_num_units, slice_dimension = 2, name="L2", activation=selu.selu)
                self.layers.append(self.L2)

                self.L2_flattened = tf.reshape(self.L2, shape=(tf.shape(self.L2)[0], self.L2_num_units * self.LSTM2_num_units * 2), name="L2_flattened")
                self.layers.append(self.L2_flattened)

                # Dense layer 3
                self.L3 = tf.layers.dense(self.L2_flattened, units=self.L3_num_units, name="L3", activation=selu.selu)
                self.layers.append(self.L3)

                self.L3_dropout = selu.dropout_selu(self.L3, self.L3_dropout_rate_placeholder, training=self.phase_placeholder, name='L3_dropout')
                self.layers.append(self.L3_dropout)

                self.core_final_layer = self.L3_dropout

            # Output layer
            with tf.variable_scope("Prediction"):
                self.Y_base_change_logits = tf.layers.dense(inputs=self.core_final_layer, units=self.output_base_change_shape, activation=selu.selu, name='Y_base_change_logits')
                self.Y_base_change = tf.nn.softmax(self.Y_base_change_logits, name='Y_base_change')
                self.layers.append(self.Y_base_change)

                self.Y_zygosity_logits = tf.layers.dense(inputs=self.core_final_layer, units=self.output_zygosity_shape, activation=selu.selu, name='Y_zygosity_logits')
                self.Y_zygosity = tf.nn.softmax(self.Y_zygosity_logits, name='Y_zygosity')
                self.layers.append(self.Y_zygosity)

                if "legacy_0.1" in self.structure:
                    self.Y_variant_type_logits = tf.layers.dense(inputs=self.core_final_layer, units=self.output_variant_type_shape, activation=selu.selu, name='Y_variant_logits')
                    self.Y_variant_type = tf.nn.softmax(self.Y_variant_type_logits, name='Y_variant')
                else:
                    self.Y_variant_type_logits = tf.layers.dense(inputs=self.core_final_layer, units=self.output_variant_type_shape, activation=selu.selu, name='Y_variant_type_logits')
                    self.Y_variant_type = tf.nn.softmax(self.Y_variant_type_logits, name='Y_variant_type')
                self.layers.append(self.Y_variant_type)

                self.Y_indel_length_logits = tf.layers.dense(inputs=self.core_final_layer, units=self.output_indel_length_shape, activation=selu.selu, name='Y_indel_length_logits')
                self.Y_indel_length = tf.nn.softmax(self.Y_indel_length_logits, name='Y_indel_length')
                self.layers.append(self.Y_indel_length)

                self.Y = [self.Y_base_change, self.Y_zygosity, self.Y_variant_type, self.Y_indel_length]

            # Extract the truth labels by output ratios
            with tf.variable_scope("Loss"):
                Y_base_change_label, Y_zygosity_label, Y_variant_type_label, Y_indel_length_label = tf.split(self.Y_placeholder, self.output_label_split, axis=1, name="label_split") 

                # Cross Entropy loss
                if "legacy_0.1" in self.structure:
                    Y_variant_type_str = "Y_variant"
                else:
                    Y_variant_type_str = "Y_variant_type"

                if not self.output_weight_enabled:
                    self.Y_base_change_cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.Y_base_change_logits, labels=Y_base_change_label, name="Y_base_change_cross_entropy")
                    self.Y_zygosity_cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.Y_zygosity_logits, labels=Y_zygosity_label, name="Y_zygosity_cross_entropy")
                    self.Y_variant_type_cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.Y_variant_type_logits, labels=Y_variant_type_label, name=Y_variant_type_str + "_cross_entropy")
                    self.Y_indel_length_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.Y_indel_length_logits, labels=Y_indel_length_label, name="Y_indel_length_entropy")
                else:
                    self.Y_base_change_cross_entropy = Qianliyan.weighted_cross_entropy(softmax_prediction=self.Y_base_change, labels=Y_base_change_label, 
                        weights=self.output_base_change_entropy_weights_placeholder, epsilon=self.epsilon, name="Y_base_change_cross_entropy")
                    self.Y_zygosity_cross_entropy = Qianliyan.weighted_cross_entropy(softmax_prediction=self.Y_zygosity, labels=Y_zygosity_label, 
                        weights=self.output_zygosity_entropy_weights_placeholder, epsilon=self.epsilon, name="Y_zygosity_cross_entropy")
                    self.Y_variant_type_cross_entropy = Qianliyan.weighted_cross_entropy(softmax_prediction=self.Y_variant_type, labels=Y_variant_type_label, 
                        weights=self.output_variant_type_entropy_weights_placeholder, epsilon=self.epsilon, name=Y_variant_type_str + "_cross_entropy")
                    self.Y_indel_length_entropy = Qianliyan.weighted_cross_entropy(softmax_prediction=self.Y_indel_length, labels=Y_indel_length_label, 
                        weights=self.output_indel_length_entropy_weights_placeholder, epsilon=self.epsilon, name="Y_indel_length_entropy")

                self.Y_base_change_loss = tf.reduce_sum(self.Y_base_change_cross_entropy, name="Y_base_change_loss")
                self.Y_zygosity_loss = tf.reduce_sum(self.Y_zygosity_cross_entropy, name="Y_zygosity_loss")
                self.Y_variant_type_loss = tf.reduce_sum(self.Y_variant_type_cross_entropy, name= Y_variant_type_str + "_loss")
                self.Y_indel_length_loss = tf.reduce_sum(self.Y_indel_length_entropy, name="Y_indel_length_loss")

                self.regularization_L2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name ]) * self.regularization_L2_lambda_placeholder

                # Weighted avergae of losses, speicified by loss_ratio
                # self.total_loss = self.Y_base_change_loss + self.Y_zygosity_loss + self.Y_variant_type_loss + self.Y_indel_length_loss
                self.total_loss = tf.reduce_sum(tf.multiply(self.task_loss_weights_placeholder, tf.stack([self.Y_base_change_loss, self.Y_zygosity_loss, self.Y_variant_type_loss, self.Y_indel_length_loss, self.regularization_L2_loss])), name="Total_loss")
            
            self.saver = tf.train.Saver(max_to_keep=1000000,)

            if "RNN" in self.structure or "LSTM" in self.structure:
                with tf.variable_scope("Training_Operation"):
                    self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_placeholder)
                    gradients = self.optimizer.compute_gradients(self.total_loss)
                    clipped_gradients = [(tf.clip_by_value(grad, -9., 9.), var) for grad, var in gradients]
                    self.training_op = self.optimizer.apply_gradients(clipped_gradients)
            else:
                self.training_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate_placeholder).minimize(self.total_loss)
            self.init_op = tf.global_variables_initializer()

            # Summary logging

            self.training_summary_op = tf.summary.merge([
                tf.summary.scalar('learning_rate', self.learning_rate_placeholder),
                tf.summary.scalar('l2_Lambda', self.regularization_L2_lambda_placeholder),
                tf.summary.scalar("Y_base_change_loss", self.Y_base_change_loss),
                tf.summary.scalar("Y_zygosity_loss", self.Y_zygosity_loss),
                tf.summary.scalar("Y_variant_type_loss", self.Y_variant_type_loss),
                tf.summary.scalar("Y_indel_length_loss", self.Y_indel_length_loss),
                tf.summary.scalar("Regularization_loss", self.regularization_L2_loss),
                tf.summary.scalar("Total_loss", self.total_loss)
            ])

            # For report or debug. Fetching histogram summary is slow, GPU utilization will be low if enabled.
            #for var in tf.trainable_variables():
            #    tf.summary.histogram(var.op.name, var)
            # self.merged_summary_op = tf.summary.merge_all()

            # Aliasing
            self.XPH = self.X_placeholder
            self.YPH = self.Y_placeholder
            self.learningRatePH = self.learning_rate_placeholder
            self.phasePH = self.phase_placeholder
            self.l2RegularizationLambdaPH = self.regularization_L2_lambda_placeholder

            # self.dropoutRateFC4PH = self.L3_dropout_rate_placeholder
            # self.dropoutRateFC5PH = tf.placeholder(tf.float32, shape=[], name='Aliasing2')
            self.loss = self.total_loss

            # Getting the total number of traininable parameters
            
            total_parameters = 0
            for variable in tf.trainable_variables():
                # shape is an array of tf.Dimension
                shape = variable.get_shape()
                # print(len(shape))
                variable_parameters = 1
                try:
                    for dim in shape:
                        # print(dim)
                        variable_parameters *= dim.value
                    total_parameters += variable_parameters
                except ValueError as ve:
                    print("Variable {:s} has unknown shape.".format(variable.name))
                    print(ve.message)
                # print(variable_parameters)
                    
            print("Total Trainable Parameters: " + str(total_parameters))

    def init(self):
        self.current_summary_writer = tf.summary.FileWriter('logs', self.session.graph)
        self.session.run(self.init_op)

    def get_summary_op_factory(self, render_function, name="Render", *args_factory, **kwargs_factory):
        """
        (Experimental)
        Wrap the rendering function as a tensorflow operation
        """

        def _get_tensor_render_op(in_tensor, *args_func, **kwargs_func):
            
            def _render_function_wrap(in_tensor, *args):
                img_arrays = [render_function(matrix, *args, **kwargs_func) for matrix in in_tensor]
                return np.stack(img_arrays, axis=0)
                # img_array = render_function(*args, **kwargs_func)
                # return np.expand_dims(img_array, axis=0)
            return tf.py_func(_render_function_wrap, [in_tensor] + list(args_func), Tout=tf.uint8, name="Plot")
             
        def _summary_op_factory(summary_name, in_tensor, *args_call, **kwargs_call):
            tf_render_op = _get_tensor_render_op(in_tensor, *args_call, **kwargs_call)

            # with tf.name_scope("Batch_Plot"):
            #     unstack_layer = tf.unstack(in_tensor, name='unstack')
            #     tensor_render_ops = [_get_tensor_render_op(slice_tensor, *args_call, **kwargs_call) for slice_tensor in unstack_layer]
            #     image_stacked = tf.stack(tensor_render_ops, name='stack_images')

            return tf.summary.image(summary_name, tf_render_op,
                                max_outputs=kwargs_call.pop('max_outputs', 3),
                                collections=kwargs_call.pop('collections', None),
                                )
        return _summary_op_factory

    @staticmethod
    def matrix_to_heatmap_encoded(matrix, cmap=plt.cm.bwr, **kwargs):
        """
        Plot the matrix by MatPlotLib matshow() function. Automatic ticks and enlargement are added with colorbar.
        **kwargs will be passed to the matshow function.

        return: (w, h, encoded_string)
            w: width of the image
            h: height of the image
            encoded_string: The image encoded in string format
        """
        # return np.array(np.expand_dims(matrix, axis=-1), dtype=np.uint8)
        matrix = matrix.transpose()
        x_ticker = mticker.MaxNLocator(nbins=9, steps=[1, 2, 5, 10], integer=True, symmetric=True, prune='both')
        x_max = matrix.shape[1]
        x_tick_values = list(map(lambda x : x + (x_max - 1) // 2, x_ticker.tick_values((-1) * (x_max - 1) // 2, (x_max - 1) // 2))) + [0, x_max - 1]
        plt.axis('off')
        matrix_aspect_ratio = float(matrix.shape[1]) / matrix.shape[0]
        if matrix_aspect_ratio > 30:
            height = 1
            width = int(min(matrix_aspect_ratio * height + 1, 200))
        elif matrix_aspect_ratio < 1 / 30:
            width = 10
            height = int(min(width / matrix_aspect_ratio + 1, 200))
        else:
            height = 4
            width = int(min(matrix_aspect_ratio * height + 1, 200))
        f = plt.figure(figsize=(width,height))
        f.tight_layout()
        ax = f.add_subplot(1, 1, 1)
        im = ax.matshow(matrix, cmap=cmap, **kwargs)
        ax.xaxis.set_ticks(x_tick_values)
        ax.yaxis.set_ticks([])
        f.colorbar(im)
        f.canvas.draw()
        bytes_io = io.BytesIO()
        f.savefig(bytes_io, format='png', transparent=True, bbox_inches='tight')
        w, h = f.canvas.get_width_height()
        encoded_string = bytes_io.getvalue()
        plt.close(f)
        bytes_io.close()
        return w, h, encoded_string

    @staticmethod
    def matrix_to_heatmap_encoded_fast(matrix, colors=[COLORS_RGB['BLUE'], COLORS_RGB['WHITE'], COLORS_RGB['RED']], enlarge_factor=1, auto_adjust_factor=5, **kwargs):
        """
        Plot the matrix by averaging 3 color RGB values according to range.
        
        colors: A list of RGB values, organized as [RGB1, RGB2, RGB3], with each RGB being a list of 3 integers
        enlarge_factor: Specify the basic number of pixels per value
        auto_adjust_factor: Autoadjust matrices which are too large in one or more of the dimensions

        return: (w, h, encoded_string)
            w: width of the image
            h: height of the image
            encoded_string: The image encoded in string format
        """
        matrix = matrix.transpose()
        vmax = np.amax(matrix) if 'vmax' not in kwargs else kwargs['vmax']
        vmin = np.amin(matrix) if 'vmin' not in kwargs else kwargs['vmin']
        M = (matrix - vmin) / (vmax - vmin)

        rgb_matrix = np.array(np.stack([np.where(M < 0.5, (1 - M * 2) * cmin + M * 2 * c0, (1 - (M - 0.5) * 2) * c0 + (M - 0.5) * 2 * cmax) for cmin, c0, cmax in zip(*colors)], axis=2) * 255, dtype=np.uint8)

        if auto_adjust_factor > 0:
            if matrix.shape[0] > 50 or matrix.shape[1] > 50:
                enlarge_factor = enlarge_factor * auto_adjust_factor
            # enlarge_factor = enlarge_factor * int(int(min(max(matrix.shape[0] / auto_adjust_factor, matrix.shape[1] / auto_adjust_factor), 10)))
        if enlarge_factor > 1:
            rgb_matrix = np.repeat(np.repeat(rgb_matrix, enlarge_factor, axis=0), enlarge_factor, axis=1)
        
        bytes_io = io.BytesIO()
        imageio.imwrite(bytes_io, rgb_matrix, format="png")
        encoded_string = bytes_io.getvalue()
        bytes_io.close()
        w, h = matrix.shape
        return w, h, encoded_string

    def get_batch_image_summary(self, batch_tensor_3D, summary_name="Plot", batch_item_suffixes=None, max_plot_in_batch=5, 
                                as_producer=False, output_task_queue=None, fast_plotting=False, **kwargs):
        """
        Plot images as tensorflow summaries.
        
        batch_tensor_3D: The tensor to be plotted, must have 3 dimensions, the first being the batch dimension
        summary_name: The name of the summary class
        batch_item_suffixes: The suffixes (tags) which will be appended to each item in the batch with the same index
        max_plot_in_batch: The maximum number of plots in this batch

        fast_plotting: Use numpy minimalistic plotting method

        as_producer: Use the function as a task producer, which puts tasks on the output_task_queue for consumption, 
            if True: return True for sucess
            if False: return a list of tf summary created by tf.Summary.Value
        output_task_queue: Used when as_producer is True, the queue where the tasks should be put
        """
        if max_plot_in_batch < 0:
            max_plot_in_batch = len(batch_tensor_3D) 
        updated_kwargs = kwargs.copy()
        if "vmax" not in updated_kwargs:
            updated_kwargs["vmax"] = np.amax(batch_tensor_3D)
        if "vmin" not in updated_kwargs:
            updated_kwargs["vmin"] = np.amin(batch_tensor_3D)

        if batch_item_suffixes is None:
            batch_item_suffixes = [str(i) for i in range(max_plot_in_batch)]
        if len(batch_item_suffixes) < max_plot_in_batch:
            batch_item_suffixes = ["".join([batch_item_suffixes[i % len(batch_item_suffixes)], "_", str(int(i / len(batch_item_suffixes)))]) for i in range(max_plot_in_batch)]
        
        if as_producer:
            
            if fast_plotting:
                plt_kwargs = dict()
                output_task_queue.put((
                    (summary_name, "color_bar"), 
                    (1, np.expand_dims(np.array([updated_kwargs["vmax"], updated_kwargs["vmin"]], dtype=float), axis=-1), ), 
                    plt_kwargs)
                )

            for item_suffix, batch_matrix in zip(batch_item_suffixes, batch_tensor_3D[:max_plot_in_batch]):
                if fast_plotting:
                    output_task_queue.put((
                        (summary_name, item_suffix),
                        (0, batch_matrix, ), 
                        updated_kwargs)
                    )
                else:
                    output_task_queue.put((
                        (summary_name, item_suffix), 
                        # Qianliyan.matrix_to_heatmap_encoded, 
                        (batch_matrix, ), 
                        updated_kwargs)
                    )
            
            return True

        else:
            image_summaries_group = []
            if fast_plotting:
                plt_kwargs = dict()
                w, h, encoded_string = Qianliyan.matrix_to_heatmap_encoded(np.expand_dims(np.array([updated_kwargs["vmax"], updated_kwargs["vmin"]], dtype=float), axis=-1), **plt_kwargs)
                img_sum = tf.Summary.Image(encoded_image_string=encoded_string, height=h, width=w)
                image_summaries_group.append(tf.Summary.Value(tag='%s/%s' % (summary_name, "color_bar"),image=img_sum))

            for item_suffix, batch_matrix in zip(batch_item_suffixes, batch_tensor_3D[:max_plot_in_batch]):
                if fast_plotting:
                    w, h, encoded_string = Qianliyan.matrix_to_heatmap_encoded_fast(batch_matrix, **updated_kwargs)
                else:
                    w, h, encoded_string = Qianliyan.matrix_to_heatmap_encoded(batch_matrix, **updated_kwargs)
                img_sum = tf.Summary.Image(encoded_image_string=encoded_string, height=h, width=w)
                image_summaries_group.append(tf.Summary.Value(tag='%s/%s' % (summary_name, item_suffix),image=img_sum))
            return image_summaries_group

    def batch_image_summary_regroup(self, return_dict):
        """
        Regroup the summaries accodring to the names
        """
        image_summaries_groups = defaultdict(list)
        for key in return_dict:
        # while not result_queue.empty():
            identity, result = return_dict[key]
            # identity, result = result_queue.get()
            # print(identity, result_queue.qsize())
            summary_name, item_suffix = identity
            w, h, encoded_string = result
            img_sum = tf.Summary.Image(encoded_image_string=encoded_string, height=h, width=w)
            image_summaries_groups[summary_name].append(tf.Summary.Value(tag='%s/%s' % (summary_name, item_suffix),image=img_sum))
        return [image_summaries_groups[key] for key in image_summaries_groups]

    @staticmethod
    def recursive_process_tensor(tensor, apply_function, recursion_text="", target_ndim=2, last_first=False, sparator="-", *args, **kwargs):
        if tensor.ndim <= target_ndim:
            return [apply_function(tensor, recursion_text, *args, **kwargs)]
        else:
            if last_first:
                rolled_tensor = np.rollaxis(tensor, -1)
            recursion_text += sparator
            processed = [Qianliyan.recursive_process_tensor(subtensor, apply_function, recursion_text + str(i), target_ndim=target_ndim, last_first=last_first, *args, **kwargs) 
                            for i, subtensor in enumerate(rolled_tensor)]
            return [item for sublist in processed for item in sublist]
            
    def get_activation_summary(self, batchX, batchY=None, operations=None, batch_item_suffixes=None, max_plot_in_batch=5, verbose=True, parallel_level=2, num_workers=8, fast_plotting=False):
        if max_plot_in_batch < 0:
            max_plot_in_batch = len(batchX)
        transformed_batch_X, transformed_batch_Y = self.tensor_transform_function(batchX[:max_plot_in_batch], batchY[:max_plot_in_batch] if batchY is not None else batchY, "predict")
        input_dictionary = {
            self.X_placeholder:transformed_batch_X, 
            # self.Y_placeholder:transformed_batch_Y,
            self.learning_rate_placeholder:0.0,
            self.phase_placeholder:False,
            self.regularization_L2_lambda_placeholder:0.0,
            self.task_loss_weights_placeholder: self.task_loss_weights,
            self.output_base_change_entropy_weights_placeholder: self.output_base_change_entropy_weights,
            self.output_zygosity_entropy_weights_placeholder: self.output_zygosity_entropy_weights,
            self.output_variant_type_entropy_weights_placeholder: self.output_variant_type_entropy_weights,
            self.output_indel_length_entropy_weights_placeholder: self.output_indel_length_entropy_weights
        }
        input_dictionary.update(self.get_structure_dict(phase='predict'))

        summaries = []
        with self.graph.as_default():
            # summary_op_factory = self.get_summary_op_factory(Qianliyan.matrix_to_heatmap)

            tensors = self.session.run(operations, feed_dict=input_dictionary)
            if parallel_level == 0:
                for op, tensor in zip(operations, tensors):
                    if verbose:
                        print(op.name)
                    draw_param = {}
                    if "Y" in op.name:
                        draw_param['vmax'] = 1.00
                        draw_param['vmin'] = 0.00
                    other_params = dict(
                        batch_item_suffixes=batch_item_suffixes,
                        max_plot_in_batch=max_plot_in_batch, 
                        fast_plotting=fast_plotting,
                        auto_adjust_factor=5
                    )
                    draw_param.update(other_params)
                    if tensor.ndim == 3:
                        summaries.append(tf.Summary(value=self.get_batch_image_summary(tensor, op.name.replace("/", "-"), **draw_param)))
                    elif tensor.ndim == 2:
                        summaries.append(tf.Summary(value=self.get_batch_image_summary(np.expand_dims(tensor, axis=-1), op.name.replace("/", "-"), **draw_param)))
                    elif tensor.ndim == 1:
                        summaries.append(tf.Summary(value=self.get_batch_image_summary(np.expand_dims(np.expand_dims(tensor, axis=-1), axis=-1), op.name.replace("/", "-"), **draw_param)))
                    else:
                        if verbose:
                            print(op.name + " has more then 3 dimensions.")
                        summary_group_list = Qianliyan.recursive_process_tensor(tensor, self.get_batch_image_summary, op.name.replace("/", "-"), target_ndim=3, last_first=True, **draw_param)
                        summary_list = [tf.Summary(value=summary_group) for summary_group in summary_group_list]
                        summaries.extend(summary_list)

            elif parallel_level == 2 and not fast_plotting:
                task_queue = multiprocessing.JoinableQueue()

                manager = multiprocessing.Manager()
                return_dict = manager.dict()
                # result_queue = multiprocessing.Queue()

                workers = [FunctionCallConsumer(Qianliyan.matrix_to_heatmap_encoded, task_queue, return_dict, name=str(i), verbose=verbose) for i in range(num_workers)]
                try:
                    for w in workers:
                        w.start()

                    for op, tensor in zip(operations, tensors):
                        if verbose:
                            print(op.name)
                        draw_param = {}
                        if "Y" in op.name:
                            draw_param['vmax'] = 1.00
                            draw_param['vmin'] = 0.00
                        other_params = dict(
                            batch_item_suffixes=batch_item_suffixes, 
                            max_plot_in_batch=max_plot_in_batch, 
                            as_producer=True, 
                            output_task_queue=task_queue,
                        )
                        draw_param.update(other_params)

                        if tensor.ndim == 3:
                            self.get_batch_image_summary(tensor, op.name.replace("/", "-"), **draw_param)
                        elif tensor.ndim == 2:
                            self.get_batch_image_summary(np.expand_dims(tensor, axis=-1), op.name.replace("/", "-"), **draw_param)
                        elif tensor.ndim == 1:
                            self.get_batch_image_summary(np.expand_dims(np.expand_dims(tensor, axis=-1), axis=-1), op.name.replace("/", "-"), **draw_param)
                        else:
                            if verbose:
                                print(op.name + " has more then 3 dimensions.")
                            Qianliyan.recursive_process_tensor(tensor, self.get_batch_image_summary, op.name.replace("/", "-"), target_ndim=3, last_first=True,**draw_param)

                    # Add a poison pill for each consumer
                    for i in range(num_workers):
                        task_queue.put(None)
                    # Wait for all of the tasks to finish
                    task_queue.join()
                    for w in workers:
                        w.join()
                    if verbose:
                        print("All workers finished.")
                    summary_group_list = self.batch_image_summary_regroup(dict(return_dict))

                    summaries = [tf.Summary(value=summary_group) for summary_group in summary_group_list]
                except KeyboardInterrupt:
                    for w in workers:
                        w.terminate()
                        w.join()
                    print("All workers terminated")
                    raise KeyboardInterrupt

            elif parallel_level == 2 and fast_plotting:
                task_queue = multiprocessing.JoinableQueue()

                manager = multiprocessing.Manager()
                return_dict = manager.dict()
                # result_queue = multiprocessing.Queue()

                def function_select_wrap(instruction, *args, **kwargs):
                    if instruction == 0:
                        return Qianliyan.matrix_to_heatmap_encoded_fast(*args, **kwargs)
                    else:
                        return Qianliyan.matrix_to_heatmap_encoded(*args, **kwargs)

                workers = [FunctionCallConsumer(function_select_wrap, task_queue, return_dict, name=str(i), verbose=verbose) for i in range(num_workers)]
                try:
                    for w in workers:
                        w.start()

                    for op, tensor in zip(operations, tensors):
                        if verbose:
                            print(op.name)
                        draw_param = {}
                        if "Y" in op.name:
                            draw_param["vmax"] = 1.00
                            draw_param["vmin"] = 0.00
                        other_params = dict(
                            batch_item_suffixes=batch_item_suffixes, 
                            max_plot_in_batch=max_plot_in_batch, 
                            as_producer=True, 
                            output_task_queue=task_queue, 
                            fast_plotting=True,
                            auto_adjust_factor=5
                        )
                        draw_param.update(other_params)

                        
                        if tensor.ndim == 3:
                            self.get_batch_image_summary(tensor, op.name.replace("/", "-"), **draw_param)
                        elif tensor.ndim == 2:
                            self.get_batch_image_summary(np.expand_dims(tensor, axis=-1), op.name.replace("/", "-"), **draw_param)
                        elif tensor.ndim == 1:
                            self.get_batch_image_summary(np.expand_dims(np.expand_dims(tensor, axis=-1), axis=-1), op.name.replace("/", "-"), **draw_param)
                        else:
                            if verbose:
                                print(op.name + " has more then 3 dimensions.")
                            Qianliyan.recursive_process_tensor(tensor, self.get_batch_image_summary, op.name.replace("/", "-"), target_ndim=3, last_first=True, **draw_param)

                    # Add a poison pill for each consumer
                    for i in range(num_workers):
                        task_queue.put(None)
                    # Wait for all of the tasks to finish
                    task_queue.join()
                    for w in workers:
                        w.join()
                    if verbose:
                        print("All workers finished.")
                    summary_group_list = self.batch_image_summary_regroup(dict(return_dict))

                    summaries = [tf.Summary(value=summary_group) for summary_group in summary_group_list]
                except KeyboardInterrupt:
                    for w in workers:
                        w.terminate()
                        w.join()
                    print("All workers terminated")
                    raise KeyboardInterrupt

        return summaries

    def test_run(self, num_samples=10, log_dir='logs'):
        writer = self.get_summary_file_writer(log_dir)
        self.session.run(self.init_op)
        dimension_X = (num_samples, self.input_shape[0], self.input_shape[1], self.input_shape[2])
        dimension_Y = (num_samples, sum(self.output_label_split))

        random_X = np.random.random(size=dimension_X)
        random_Y = np.random.random(size=dimension_Y)

        print("Running test for training")
        result = self.train(random_X, random_Y)
        print("Training test finished, results:", result)
        print("Running test for prediction")
        result = self.predict(random_X)
        print("Prediction test finished, results:", result)
        print("Running test for loss")
        result = self.get_loss(random_X, random_Y)
        print("Loss test finished, results:", result)
        summaries = self.get_activation_summary(random_X, random_Y, operations=self.layers, 
            batch_item_suffixes=["A", "B", "C", "D", "E", "F"], parallel_level=0, num_workers=5, fast_plotting=True)
        for summary in summaries:
            writer.add_summary(summary)
        writer.close()
        print("All test finished")

    def close(self):
        self.session.close()

    def crop_X_batch_at_center(self, batchX, shape):
        if batchX.shape[1] != shape[1]:
            start_index = batchX.shape[1] // 2 - shape[1] // 2
            batchX = batchX[:, start_index:start_index + shape[1], :, :]
        return batchX

    def train(self, batchX, batchY, result_caching=False):
        #for i in range(len(batchX)):
        #    tf.image.per_image_standardization(batchX[i])
        transformed_batch_X, transformed_batch_Y = self.tensor_transform_function(batchX, batchY, "train")
        input_dictionary = {
            self.X_placeholder:transformed_batch_X, 
            self.Y_placeholder:transformed_batch_Y,
            self.learning_rate_placeholder:self.learning_rate_value,
            self.phase_placeholder:True,
            self.regularization_L2_lambda_placeholder:self.l2_regularization_lambda_value,
            self.task_loss_weights_placeholder: self.task_loss_weights,
            self.output_base_change_entropy_weights_placeholder: self.output_base_change_entropy_weights,
            self.output_zygosity_entropy_weights_placeholder: self.output_zygosity_entropy_weights,
            self.output_variant_type_entropy_weights_placeholder: self.output_variant_type_entropy_weights,
            self.output_indel_length_entropy_weights_placeholder: self.output_indel_length_entropy_weights
        }
        input_dictionary.update(self.get_structure_dict(phase='train'))
        loss, _, summary = self.session.run( (self.loss, self.training_op, self.training_summary_op),
                                              feed_dict=input_dictionary)
        if result_caching:
            self.output_cache['training_loss'] = loss
            self.output_cache['training_summary'] = summary

            # Aliasing
            self.trainLossRTVal = self.output_cache['training_loss']
            self.trainSummaryRTVal = self.output_cache['training_summary']
        return loss, summary

    def predict(self, XArray, result_caching=False):
        #for i in range(len(batchX)):
        #    tf.image.per_image_standardization(XArray[i])
        transformed_batch_X, _ = self.tensor_transform_function(XArray, None, "predict")
        # XArray = self.crop_X_batch_at_center(XArray, self.input_shape_tf)
        input_dictionary = {
            self.X_placeholder:transformed_batch_X,
            self.learning_rate_placeholder:0.0,
            self.phase_placeholder:False,
            self.regularization_L2_lambda_placeholder:0.0
        }
        input_dictionary.update(self.get_structure_dict(phase='predict'))
        
        base, zygosity, variant_type, indel_length = self.session.run( self.Y, feed_dict=input_dictionary)

        if result_caching:
            self.output_cache['prediction_base'] = base
            self.output_cache['prediction_zygosity'] = zygosity
            self.output_cache['prediction_variant_type'] = variant_type
            self.output_cache['prediction_indel_length'] = indel_length

            # Aliasing
            self.predictBaseRTVal = self.output_cache['prediction_base']
            self.predictZygosityRTVal = self.output_cache['prediction_zygosity']
            self.predictVarTypeRTVal = self.output_cache['prediction_variant_type']
            self.predictIndelLengthRTVal = self.output_cache['prediction_indel_length']

        return base, zygosity, variant_type, indel_length

    def get_loss(self, batchX, batchY, result_caching=False):
        #for i in range(len(batchX)):
        #    tf.image.per_image_standardization(batchX[i])
        # batchX = self.crop_X_batch_at_center(batchX, self.input_shape_tf)
        transformed_batch_X, transformed_batch_Y = self.tensor_transform_function(batchX, batchY, "predict")
        input_dictionary = {
            self.X_placeholder:transformed_batch_X, 
            self.Y_placeholder:transformed_batch_Y,
            self.learning_rate_placeholder:0.0,
            self.phase_placeholder:False,
            self.regularization_L2_lambda_placeholder:0.0,
            self.task_loss_weights_placeholder: self.task_loss_weights,
            self.output_base_change_entropy_weights_placeholder: self.output_base_change_entropy_weights,
            self.output_zygosity_entropy_weights_placeholder: self.output_zygosity_entropy_weights,
            self.output_variant_type_entropy_weights_placeholder: self.output_variant_type_entropy_weights,
            self.output_indel_length_entropy_weights_placeholder: self.output_indel_length_entropy_weights
        }
        input_dictionary.update(self.get_structure_dict(phase='predict'))

        loss = self.session.run( self.loss, feed_dict=input_dictionary)
        if result_caching:
            self.output_cache['prediction_loss'] = loss

            # Aliasing
            self.getLossLossRTVal = self.output_cache['prediction_loss']
        return loss


    def save_parameters(self, file_name):
        # with self.g.as_default():
        #     self.saver = tf.train.Saver()
        self.saver.save(self.session, file_name)

    def restore_parameters(self, file_name):
        # self.session =  tf.Session()
        # new_saver = tf.train.import_meta_graph(file_name + '.meta')
        # new_saver.restore(self.session, file_name)            

        # with self.g.as_default():
        #     self.saver = tf.train.Saver()
        self.saver.restore(self.session, file_name)

    def get_variable_objects(self, regular_expression):
        regex = re.compile(regular_expression)
        variable_list = []
        with self.graph.as_default():
            for variable in tf.trainable_variables():
                if regex.match(variable.name):
                    variable_list.append(variable)
        return variable_list

    def get_operation_objects(self, regular_expression, exclude_expression=".*(grad|tags|Adam).*"):
        regex = re.compile(regular_expression)
        regex_exclude = re.compile(exclude_expression)
        operation_list = []
        
        for op in self.graph.get_operations():
            if regex.match(op.name) and not regex_exclude.match(op.name):
                print(op.name)
                operation_list.append(op)
        return operation_list

    def get_summary_file_writer(self, logs_path):
        if hasattr(self, "current_summary_writer"):
            self.current_summary_writer.close()
        self.current_summary_writer = tf.summary.FileWriter(logs_path, graph=self.graph)
        return self.current_summary_writer

    def set_task_loss_weights(self, task_loss_weights=[1, 1, 1, 1, 1]):
        self.task_loss_weights = np.array(task_loss_weights, dtype=float)

    def set_learning_rate(self, learning_rate):
        self.learning_rate_value = learning_rate
        return self.learning_rate_value

    def decay_learning_rate(self):
        self.learning_rate_value = self.learning_rate_value * self.learning_rate_decay_rate
        return self.learning_rate_value

    def set_l2_regularization_lambda(self, l2_regularization_lambda):
        self.l2_regularization_lambda_value = l2_regularization_lambda
        return self.l2_regularization_lambda_value

    def decay_l2_regularization_lambda(self):
        self.l2_regularization_lambda_value = self.l2_regularization_lambda_value * self.l2_regularization_lambda_decay_rate
        return self.l2_regularization_lambda_value
    
    def pretty_print_variables(self, regular_expression):
        variable_list = self.get_variable_objects(regular_expression)
        result_string_list = []
        for v in variable_list:
            variable_value = self.session.run(v)
            result_string_list.append(v.name)
            result_string_list.append(Qianliyan.pretty_print_np_tensor(variable_value) + '\n')
        return '\n'.join(result_string_list)

    @staticmethod
    def pretty_print_np_tensor(tensor, element_separator='\t'):
        if tensor.ndim == 1:
            return element_separator.join( ('%.16f') % value for value in tensor)
        elif tensor.ndim == 2:
            return_list = []
            for row in tensor:
                return_list.append(Qianliyan.pretty_print_np_tensor(row, element_separator=element_separator))
            return '\n'.join(return_list)
        else:
            return_list = []
            for sub_tensor in tensor:
                return_list.append('[\n' + Qianliyan.pretty_print_np_tensor(sub_tensor, element_separator=element_separator) + '\n]')
            return '\n'.join(return_list)

    def __del__(self):
        if hasattr(self, "current_summary_writer"):
            self.current_summary_writer.close()
        self.session.close()

    # Aliasing, Backward Compatibility

    getLoss = get_loss
    saveParameters = save_parameters
    restoreParameters = restore_parameters
    summaryFileWriter = get_summary_file_writer

    def trainNoRT(self, batchX, batchY):
        self.train(batchX, batchY, result_caching=True)
    
    def predictNoRT(self, XArray, output_activation=False):
        if output_activation:
            summaries = self.get_activation_summary(XArray)
            for summary in summaries:
                self.current_summary_writer.add_summary(summary)
        self.predict(XArray, result_caching=True)

    def getLossNoRT(self, batchX, batchY):
        self.get_loss(batchX, batchY, result_caching=True)

    def setLearningRate(self, learningRate=None):
        if learningRate == None:
            new_learning_rate = self.decay_learning_rate()
        else:
            new_learning_rate = self.set_learning_rate(learningRate)
        return new_learning_rate

    def setL2RegularizationLambda(self, l2RegularizationLambda=None):
        if  l2RegularizationLambda == None:
            new_l2_regularization_lambda = self.decay_l2_regularization_lambda()
        else:
            new_l2_regularization_lambda = self.set_l2_regularization_lambda(l2RegularizationLambda)
        return new_l2_regularization_lambda


class FunctionCallConsumer(multiprocessing.Process):
    def __init__(self, target_function, task_queue, result_queue, name="c", verbose=False):
        multiprocessing.Process.__init__(self)
        self.target_function = target_function
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.name = name
        self.verbose = verbose

    def run(self):
        if self.verbose:
            print("Consumer {:s} is starting.".format(self.name))
        while True:
            next_task = self.task_queue.get()
            if next_task is None:
                self.task_queue.task_done()
                break

            # identity, f, args, kwargs = next_task["identity"], next_task["f"], next_task["args"], next_task["kwargs"]
            # identity, f, args, kwargs = next_task
            identity, args, kwargs = next_task
            # answer = f(*args, **kwargs)
            answer = self.target_function(*args, **kwargs)
            self.task_queue.task_done()
            # self.result_queue.put((identity, answer))
            self.result_queue[identity] = (identity, answer)
            if self.verbose:
                print("Consumer {:s} finished".format(self.name), identity)
        if self.verbose:
            print("Consumer {:s} is terminating.".format(self.name))
        return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="Qianliyan (Variant caller)")

    parser.add_argument('-t', '--test', action='store_true',
            help="Run a unit test")

    parser.add_argument('-v', '--variables', type=str, default=None,
            help="Print variables matching the regular expression. default: %(default)s")

    parser.add_argument('-m', '--model', type=str, default=None,
            help="The path to the model to be restored. default: %(default)s")

    parser.add_argument('-l', '--log_dir', type=str, default="logs",
            help="The path to the log directory. default: %(default)s")

    args = parser.parse_args()
    if args.test:
        print("Running unit test")
        q = Qianliyan()
        if args.model is not None:
            q.restoreParameters(os.path.abspath(args.model))
        q.test_run(log_dir=args.log_dir)
        
        # q.get_operation_objects("L1")
        # print([op.name for op in q.graph.get_operations()])
        sys.exit(0)
    
    if args.variables is not None:
        q = Qianliyan()
        q.init()
        if args.model is not None:
            q.restoreParameters(os.path.abspath(args.model))
        print(q.pretty_print_variables(args.variables))
        sys.exit(0)
