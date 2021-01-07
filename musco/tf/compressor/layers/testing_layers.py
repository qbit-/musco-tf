import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.layers.recurrent import DropoutRNNCellMixin

from musco.tf.compressor.decompositions.decomp_functions import uv_decompose
from musco.tf.compressor.layers.decomp_recurrent_cell import FusedSVDLSTMCell


def test_fuseduvlstmcell_1():
    def get_rnn_model1():
        input_x = layers.Input(shape=(28, 28))
        out = layers.Reshape([28, 28])(input_x)
        cell = tf.keras.layers.LSTMCell(
            12,
            activation='tanh',
            recurrent_activation='hard_sigmoid',
            unit_forget_bias=True,)
        rnn = layers.RNN(cell)
        out = rnn(out)
        return keras.Model(inputs=input_x, outputs=out)

    def get_uv_rnn_model1(rank, recurrent_rank):
        input_x = layers.Input(shape=(28, 28))
        out = layers.Reshape([28, 28])(input_x)
        cell = FusedSVDLSTMCell(units=12,
                               rank=rank,
                               recurrent_rank=recurrent_rank,
                               activation='tanh',
                               recurrent_activation='hard_sigmoid',
                               unit_forget_bias=True,
                               )
        rnn = layers.RNN(cell)
        out = rnn(out)
        return keras.Model(inputs=input_x, outputs=out)

    data = np.random.rand(4, 28, 28).astype(np.float32)
    model1 = get_rnn_model1()
    kernel, r_kernel, bias = model1.get_weights()

    input_dim, output_dim = kernel.shape
    uv_kernel = uv_decompose(
        kernel, max_rank=1000, epsilon=0)
    uv_r_kernel = uv_decompose(
        r_kernel, max_rank=1000, epsilon=0)
    weights = []
    for w in uv_kernel:
        weights.append(w)

    for w in uv_r_kernel:
        weights.append(w)
    weights.append(bias)

    ranks = uv_kernel[0].shape[1]
    r_ranks = uv_r_kernel[0].shape[1]

    uv_model1 = get_uv_rnn_model1(ranks, r_ranks)
    uv_model1.set_weights(weights)
    out1 = model1(data)
    uv_out1 = uv_model1(data)
    assert np.allclose(out1.numpy(), uv_out1.numpy(), atol=1e-5)
    uv_model1.save('test_save.h5')
    uv_model_res = keras.models.load_model(
        'test_save.h5', custom_objects={'FusedSVDLSTMCell': FusedSVDLSTMCell})
    uv_out1_2 = uv_model_res(data)
    assert np.allclose(out1.numpy(), uv_out1_2.numpy(), atol=1e-5)


def test_fuseduvlstmcell_2():
    def get_rnn_model2():
        input_x = layers.Input(shape=(28, 28))
        rnn = tf.keras.layers.RNN(
            tf.keras.layers.LSTMCell(
                2,
                activation='tanh',
                recurrent_activation='sigmoid',
                bias_initializer='ones'),
            return_sequences=True,
            return_state=True)
        out = rnn(input_x)
        return keras.Model(inputs=input_x, outputs=out)

    def get_uv_rnn_model2(rank, recurrent_rank):
        input_x = layers.Input(shape=(28, 28))
        cell = FusedSVDLSTMCell(units=2,
                               rank=rank, recurrent_rank=recurrent_rank,
                               activation='tanh',
                               recurrent_activation='sigmoid',
                               bias_initializer='ones')
        rnn = layers.RNN(cell,
                         return_sequences=True,
                         return_state=True)
        out = rnn(input_x)
        return keras.Model(inputs=input_x, outputs=out)

    data = np.random.rand(4, 28, 28).astype(np.float32)
    model2 = get_rnn_model2()
    kernel, r_kernel, bias = model2.get_weights()

    input_dim, output_dim = kernel.shape
    n_ker = 4

    uv_kernel = uv_decompose(
        kernel, max_rank=1000, epsilon=0)
    uv_r_kernel = uv_decompose(r_kernel, max_rank=1000, epsilon=0)
    weights = []
    for w in uv_kernel:
        weights.append(w)

    for w in uv_r_kernel:
        weights.append(w)
    weights.append(bias)

    rank = uv_kernel[0].shape[1]
    r_rank = uv_r_kernel[0].shape[1]

    uv_model2 = get_uv_rnn_model2(rank, r_rank)
    uv_model2.set_weights(weights)
    out2 = model2(data)
    uv_out2 = uv_model2(data)
    assert np.allclose(out2[0].numpy(), uv_out2[0].numpy(), atol=1e-5)
    assert np.allclose(out2[1].numpy(), uv_out2[1].numpy(), atol=1e-5)


def test_fuseduvlstmcell_3():
    def get_rnn_model1():
        input_x = layers.Input(shape=(28, 28))
        out = layers.Reshape([28, 28])(input_x)
        cell = tf.keras.layers.LSTMCell(
            12,
            activation='tanh',
            recurrent_activation='hard_sigmoid',
            unit_forget_bias=True,)
        rnn = layers.RNN(cell)
        out = rnn(out)
        return keras.Model(inputs=input_x, outputs=out)

    def get_uv_rnn_model1(rank, recurrent_rank, rnn_layer):
        input_x = layers.Input(shape=(28, 28))
        out = layers.Reshape([28, 28])(input_x)
        cell = FusedSVDLSTMCell(units=12,
                               rank=rank,
                               recurrent_rank=recurrent_rank,
                               activation='tanh',
                               recurrent_activation='hard_sigmoid',
                               unit_forget_bias=True,
                               parent_layer=rnn_layer 
                               )
        rnn = layers.RNN(cell)
        out = rnn(out)
        return keras.Model(inputs=input_x, outputs=out)

    data = np.random.rand(4, 28, 28).astype(np.float32)
    model1 = get_rnn_model1()
    kernel, r_kernel, bias = model1.get_weights()

    input_dim, output_dim = kernel.shape
    uv_kernel = uv_decompose(
        kernel, max_rank=1000, epsilon=0)
    uv_r_kernel = uv_decompose(
        r_kernel, max_rank=1000, epsilon=0)
    weights = []
    for w in uv_kernel:
        weights.append(w)

    for w in uv_r_kernel:
        weights.append(w)
    weights.append(bias)

    ranks = uv_kernel[0].shape[1]
    r_ranks = uv_r_kernel[0].shape[1]

    uv_model1 = get_uv_rnn_model1(ranks, r_ranks, model1.layers[2].cell)
    uv_model1.set_weights(weights)
    out1 = model1(data)
    uv_out1 = uv_model1(data)
    assert np.allclose(out1.numpy(), uv_out1.numpy(), atol=1e-5)
    uv_model1.save('test_save.h5')
    uv_model_res = keras.models.load_model(
        'test_save.h5', custom_objects={'FusedSVDLSTMCell': FusedSVDLSTMCell})
    uv_out1_2 = uv_model_res(data)
    assert np.allclose(out1.numpy(), uv_out1_2.numpy(), atol=1e-5)