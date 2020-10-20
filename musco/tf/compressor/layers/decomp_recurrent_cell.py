"""
Classes for decomposed LSTM cells
"""


import numpy as np
import itertools as it

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import backend as K
from tensorflow.keras import layers
from tensorflow.python.keras.layers.recurrent import DropoutRNNCellMixin

from tensorflow.python.keras import activations
from tensorflow.python.keras import initializers
from tensorflow.python.keras.utils import tf_utils

from tensorflow.python.ops import array_ops           # for initializer
from tensorflow.python.framework import tensor_shape   # for initializer
from tensorflow.python.training.tracking import data_structures  # for serialization
from tensorflow.python.util import nest              # for initializer

from musco.tf.compressor.decompositions.decomp_functions import uv_decompose

class FusedSVDLSTMCell(DropoutRNNCellMixin, layers.Layer):
    """Cell class for LSTM with skeleton decomposition.
    See [the Keras RNN API guide](https://www.tensorflow.org/guide/keras/rnn)
    for details about the usage of RNN API.
    Arguments:
      units: integer, dimensionality of the output space.
      parent_layer: LSTM or LSTMCell parent layer, which weights should decompose.
      rank: integer,
               rank of the decomposition. Default 2
      recurrent_rank: integer,
               rank of the decomposition. Default None - in this case recurrent_rank is equal to rank.
      activation: Activation function to use.
        Default: hyperbolic tangent (`tanh`).
        If you pass `None`, no activation is applied
        (ie. "linear" activation: `a(x) = x`).
      recurrent_activation: Activation function to use
        for the recurrent step.
        Default: sigmoid (`sigmoid`).
        If you pass `None`, no activation is applied
        (ie. "linear" activation: `a(x) = x`).
      use_bias: Boolean, (default `True`), whether
      the layer uses a bias vector.
      kernel_initializer: Initializer for the `kernel` weights matrix,
        used for the linear transformation of the inputs.
        Default: `glorot_uniform`
      recurrent_initializer: Initializer for the `recurrent_kernel`
        weights matrix, used for the linear transformation
        of the recurrent state.
        Default: `orthogonal`
      bias_initializer: Initializer for the bias vector. Default: `zeros`.
      unit_forget_bias: Boolean.
        If True, add 1 to the bias of the forget gate at initialization.
        Setting it to true will also force `bias_initializer="zeros"`.
        This is recommended in [Jozefowicz et
          al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
      dropout: Float between 0 and 1. Fraction of the units
               to drop for the linear
        transformation of the inputs. Default: 0.
      recurrent_dropout: Float between 0 and 1.
        Fraction of the units to drop for
        the linear transformation of the recurrent state. Default: 0.
    Call arguments:
      inputs: A 2D tensor.
      states: List of state tensors corresponding to the previous timestep.
      training: Python boolean indicating whether the layer should behave in
        training mode or in inference mode. Only relevant when `dropout` or
        `recurrent_dropout` is used.
    Examples:
    ```python
    inputs = np.random.random([32, 10, 8]).astype(np.float32)
    rnn = tf.keras.layers.RNN(FusedSVDLSTMCell(4))
    output = rnn(inputs)  # The output has shape `[32, 4]`.
    rnn = tf.keras.layers.RNN(FusedSVDLSTMCell(4),
        return_sequences=True,
        return_state=True)
    # whole_sequence_output has shape `[32, 10, 4]`.
    # final_state has shape `[32, 4]`.
    whole_sequence_output, final_state = rnn(inputs)
    ```
    """
    _counter = it.count(0)

    def __init__(self,
                 units,
                 rank=2,
                 recurrent_rank=None,
                 parent_layer=None,
                 activation='tanh',
                 recurrent_activation='sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 dropout=0.,
                 recurrent_dropout=0.,
                 **kwargs):
        self.counter = next(self._counter)
        super(FusedSVDLSTMCell, self).__init__(**kwargs)
    
        self.rank = rank
        if recurrent_rank is None:
            self.recurrent_rank = rank
        else:
            self.recurrent_rank = recurrent_rank
        
        if parent_layer is None:
            self.units = units

            self.activation = activations.get(activation)
            self.recurrent_activation = activations.get(recurrent_activation)
            self.use_bias = use_bias

            self.kernel_initializer = initializers.get(kernel_initializer)
            self.recurrent_initializer = initializers.get(recurrent_initializer)
            self.bias_initializer = initializers.get(bias_initializer)
            self.unit_forget_bias = unit_forget_bias

            self.dropout = min(1., max(0., dropout))
            self.recurrent_dropout = min(1., max(0., recurrent_dropout))
            
            self.exist_parent = False
        elif isinstance(parent_layer, keras.layers.LSTM)\
            or isinstance(parent_layer, tf.python.keras.layers.recurrent.LSTMCell)\
            or isinstance(parent_layer, keras.layers.LSTMCell):
            
            self.units = parent_layer.units

            self.activation = parent_layer.activation
            self.recurrent_activation = parent_layer.recurrent_activation
            self.use_bias = parent_layer.use_bias

            self.kernel_initializer = parent_layer.kernel_initializer
            self.recurrent_initializer = parent_layer.recurrent_initializer
            self.bias_initializer = parent_layer.bias_initializer
            self.unit_forget_bias = parent_layer.unit_forget_bias

            self.dropout = parent_layer.dropout
            self.recurrent_dropout = parent_layer.recurrent_dropout
            
            self.exist_parent = True
            
            kernel, r_kernel, bias = parent_layer.get_weights()

            input_dim, output_dim = kernel.shape
            uv_kernel = uv_decompose(
                kernel, max_rank=self.rank, epsilon=0)
            uv_r_kernel = uv_decompose(
                r_kernel, max_rank=self.recurrent_rank, epsilon=0)
            
            self.kernel_u = tf.Variable(uv_kernel[0], name='kernel_u', trainable=True)
            self.kernel_v = tf.Variable(uv_kernel[1], name='kernel_v', trainable=True)
            self.recurrent_kernel_u = tf.Variable(uv_r_kernel[0], name='recurrent_kernel_u', trainable=True)
            self.recurrent_kernel_v = tf.Variable(uv_r_kernel[1], name='recurrent_kernel_v', trainable=True)
            
            if self.use_bias:
                self.bias = tf.Variable(bias, name='bias', trainable=True)
            else:
                self.bias = None

            
        else:
            raise ValueError("Parent layer should be LSTM or LSTMCell")
        
        
        self.state_size = data_structures.NoDependency(
            [self.units, self.units])
        self.output_size = self.units

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        if not self.exist_parent:
            self.kernel_u = self.add_weight(
                shape=(input_shape[-1], self.rank),
                name='kernel_u',
                initializer=self.kernel_initializer)

            self.kernel_v = self.add_weight(
                shape=(self.rank, self.units * 4),
                name='kernel_v',
                initializer=self.kernel_initializer)

            self.recurrent_kernel_u = self.add_weight(
                shape=(self.units, self.recurrent_rank),
                name='recurrent_kernel_u',
                initializer=self.recurrent_initializer)

            self.recurrent_kernel_v = self.add_weight(
                shape=(self.recurrent_rank, self.units * 4),
                name='recurrent_kernel_v',
                initializer=self.kernel_initializer)

            if self.use_bias:
                if self.unit_forget_bias:

                    def bias_initializer(_, *args, **kwargs):
                        return tf.keras.backend.concatenate([
                            self.bias_initializer((self.units,), *args, **kwargs),
                            initializers.Ones()((self.units,), *args, **kwargs),
                            self.bias_initializer(
                                (self.units * 2,), *args, **kwargs),
                        ])
                else:
                    bias_initializer = self.bias_initializer
                self.bias = self.add_weight(
                    shape=(self.units * 4,),
                    name='bias',
                    initializer=bias_initializer)
            else:
                self.bias = None
        self.built = True

    def _compute_carry_and_output_fused(self, z, c_tm1):
        """Computes carry and output using fused kernels."""
        z0, z1, z2, z3 = z
        i = self.recurrent_activation(z0)
        f = self.recurrent_activation(z1)
        c = f * c_tm1 + i * self.activation(z2)
        o = self.recurrent_activation(z3)
        return c, o

    def call(self, inputs, states, training=None):
        h_tm1 = states[0]  # previous memory state
        c_tm1 = states[1]  # previous carry state

        dp_mask = self.get_dropout_mask_for_cell(
            inputs, training, count=4)
        rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(
            h_tm1, training, count=4)

        if 0. < self.dropout < 1.:
            inputs = inputs * dp_mask[0]
        z = K.dot(K.dot(inputs, self.kernel_u), self.kernel_v)
        z += K.dot(K.dot(h_tm1, self.recurrent_kernel_u),
                   self.recurrent_kernel_v)

        if self.use_bias:
            z = tf.nn.bias_add(z, self.bias)

        # split input along the last dimension, which is the type of
        # the kernel (input, forget, memory and output)
        z = array_ops.split(z, num_or_size_splits=4, axis=1)
        c, o = self._compute_carry_and_output_fused(z, c_tm1)

        h = o * self.activation(c)
        return h, [h, c]

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return list(_generate_zero_filled_state_for_cell(
            self, inputs, batch_size, dtype))

    def get_config(self):
        config = super(FusedSVDLSTMCell, self).get_config()
        config.update({
            'units': self.units,
            'rank': self.rank,
            'recurrent_rank': self.recurrent_rank,
            'activation': activations.serialize(self.activation),
            'recurrent_activation': activations.serialize(
                self.recurrent_activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(
                self.kernel_initializer),
            'recurrent_initializer': initializers.serialize(
                self.recurrent_initializer),
            'bias_initializer': initializers.serialize(
                self.bias_initializer),
            'unit_forget_bias': self.unit_forget_bias,
            'dropout': self.dropout,
            'recurrent_dropout': self.recurrent_dropout
        })

        return config
    
#     def load_decomposed_weights(layer, rank)


def _generate_zero_filled_state_for_cell(cell, inputs, batch_size, dtype):
    if inputs is not None:
        batch_size = array_ops.shape(inputs)[0]
        dtype = inputs.dtype
    return _generate_zero_filled_state(batch_size, cell.state_size, dtype)


def _generate_zero_filled_state(batch_size_tensor, state_size, dtype):
    """Generate a zero filled tensor with shape [batch_size, state_size]."""
    if batch_size_tensor is None or dtype is None:
        raise ValueError(
            'batch_size and dtype cannot be None while constructing '
            'initial state: '
            'batch_size={}, dtype={}'.format(batch_size_tensor, dtype))

    def create_zeros(unnested_state_size):
        flat_dims = tensor_shape.as_shape(unnested_state_size).as_list()
        init_state_size = [batch_size_tensor] + flat_dims
        return tf.zeros(init_state_size, dtype=dtype)

    if nest.is_sequence(state_size):
        return nest.map_structure(create_zeros, state_size)
    else:
        return create_zeros(state_size)
