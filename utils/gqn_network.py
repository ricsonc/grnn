import constants as const

import logging
from os.path import join

import numpy as np
import tensorflow as tf
from collections import OrderedDict
import math


def gqn2d_encoder(input_images, input_cam_posrot, is_training=None, scope='multi_camera_cnn',
                 weight_decay=0.001, reuse=False):
                
    camera_encoder = [['C', 2, 2, 32 * 8],
                      ['RELU'],
                      ['sR', 2, 2, 32 * 8]]
    camera_encoder_after_cam = [
                       ['sR', 3, 1, 32 * 8],
                       ['C', 1, 1, 32 * 8],
                       ['RELU'],
                       ['P', 16, 16]]

    num_context = len(input_images)
    with tf.variable_scope(scope, reuse=reuse):

        # prepare inputs
        stacked_input_images = tf.concat(input_images, axis=0)
        stacked_input_cam_posrot = tf.concat(input_cam_posrot, axis=0)

        inputs = stacked_input_images
        layer_id = 0
        for layer in camera_encoder:
            out = make_cnn_layer(inputs, layer,
                                         scope=f"camera_cnn/{layer_id}", reuse=reuse,
                                         is_training=is_training,
                                         weight_decay=weight_decay)
            layer_id += 1
            inputs = out
        camera_encoder_out = out
        # concat cam info
        _, h, w, c = camera_encoder_out.get_shape()
        cam_posrot = tf.expand_dims(tf.expand_dims(stacked_input_cam_posrot, 1), 1)
        cam_posrot_tiled = tf.tile(cam_posrot, [1, h, w, 1])
        camera_encoder_out = tf.concat([camera_encoder_out, cam_posrot_tiled], 3)

        inputs = camera_encoder_out
        for layer in camera_encoder_after_cam:
            out  = make_cnn_layer(inputs, layer,
                              scope=f"camera_cnn/post_cam/{layer_id}", reuse=reuse,
                              is_training=is_training,
                              weight_decay=weight_decay)
            layer_id += 1
            inputs = out
        camera_encoder_out2 = out
        camera_out = tf.split(camera_encoder_out2, num_context, axis=0)
        return camera_out

def make_cnn_layer(inputs, layer, weight_decay=0.0, is_training=None, scope='', reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        layer_type = layer[0]

        name = None
        if layer_type == 'C':
            assert(len(layer) == 4, "convolution takes 4 arguments")
            k, st, nc = layer[1:]
            out = tf.layers.conv2d(
                inputs, nc, k,
                strides=st,
                padding='valid',
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=weight_decay),
                name=f'conv',
                reuse=reuse)
        elif layer_type == 'RELU':
            out = tf.nn.relu(inputs)
        elif layer_type == 'P':
            assert(len(layer) == 3, "pool layer takes 3 arguments")
            k, st = layer[1:]
            out = tf.layers.average_pooling2d(inputs, k, st, name=f'pool')
        elif layer_type == 'sR':
            assert(len(layer) == 4, "stacked residual takes 4 arguments")
            k, st, nc = layer[1:]

            res = tf.layers.conv2d(
                inputs, int(nc), k,
                strides=st,
                activation=None,
                padding='same',
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=weight_decay),
                name=f'sR_res',
                reuse=reuse)
            conv1 =  tf.layers.conv2d(
                inputs, int(nc)//2, 3,
                strides=1,
                activation=tf.nn.relu,
                padding='same',
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=weight_decay),
                name=f'sR_conv1',
                reuse=reuse)
            conv2 =  tf.layers.conv2d(
                conv1, int(nc), k,
                strides=st,
                activation=tf.nn.relu,
                padding='same',
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=weight_decay),
                name=f'sR_conv2',
                reuse=reuse)
            out = res + conv2
        else:
            assert(1==2, f"Unknown layer_type: {layer_type}")
    return out


def compute_kl_loss(mu_q, ln_var_q, mu_p, ln_var_p):

    ln_det_q = tf.reduce_sum(ln_var_q, axis=[1, 2, 3])
    ln_det_p = tf.reduce_sum(ln_var_p, axis=[1, 2, 3])
    var_p = tf.exp(ln_var_p)
    var_q = tf.exp(ln_var_q)
    tr_qp = tf.reduce_sum(var_q / var_p, axis=[1, 2, 3])
    _, h, w, c = mu_q.get_shape()
    k = float(h.value * w.value * c.value)
    diff = mu_p - mu_q
    term2 = tf.reduce_sum(diff * diff / var_p, axis=[1, 2, 3])
    return 0.5 * (tr_qp + term2 - k + ln_det_p - ln_det_q)


def make_lstmConv(inputs, cam_posrot, output_image, layers,
                  stochastic=True, weight_decay=0.0, is_training=None, scope='',
                  reuse=False, output_debug=False):
    
    is_convLSTM_start = False
    out = inputs
    extra = dict()
    with tf.variable_scope(scope, reuse=reuse):
        for i, layer in enumerate(layers):
            layer_type = layer[0]

            if layer_type == 'convLSTM':
                assert(len(layer) == 4, "convolution takes 4 arguments")

                if not is_convLSTM_start:
                    lstm_size, n_filters, number_steps, code_size = layer[1:]
                    '''
                    lstm_size: number of channels used in lstm state/cell
                    n_filters: number of channels in output
                    number_steps: number of conv-lstm steps
                    code_size: number of channels in latent code
                    '''
                    input_shape = tf.shape(out)
                    _, h_in, w_in, c_in = out.get_shape()
                    _, h, w, c = output_image.get_shape()
                    sh = int(h)//4
                    sw = int(w)//4

                    if cam_posrot is not None:
                        lstm_v = tf.layers.conv2d_transpose(
                            cam_posrot,
                            12,
                            sh,
                            strides=(sh, sh),
                            use_bias=False,
                            padding = "same",
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=weight_decay),
                            name=f'convLSTM_v{i}')
                    else:
                        bs = inputs.shape.as_list()[0]
                        lstm_v = tf.zeros((bs, sh, sh, 0))

                        #if int(h_in) != sh: # use convolution to change its size

                    #set to True if input is B x 1 x 1 x C
                    if h_in == 1:
                        lstm_r = tf.layers.conv2d_transpose(
                            out,
                            int(c_in),
                            sh,
                            strides=(sh, sh),
                            use_bias=False,
                            padding = "same",
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=weight_decay),
                            name=f'convLSTM_r{i}')
                    else:
                        assert h_in == sh
                        lstm_r = out

                    lstm_h = tf.zeros([input_shape[0], int(h)//4, int(w)//4, lstm_size], dtype=tf.float32)
                    lstm_c = tf.zeros([input_shape[0], int(h)//4, int(w)//4, lstm_size], dtype=tf.float32)

                    #change this to accomodate a variable channel output
                    dims = 3 + const.EMBEDDING_LOSS * const.embedding_size
                    lstm_u = tf.zeros(output_image.shape.as_list()[:-1]+[dims], dtype=tf.float32)


                    g_mu_var = []
                    lstm_h_g = []
                    lstm_u_g = []
                    lstm_reuse=reuse
                    for step_id in range(number_steps):
                        if step_id > 0: lstm_reuse=True

                        lstm_input = tf.concat([lstm_h, lstm_r, lstm_v], 3)
                        lstm_u_g.append(lstm_u)
                        lstm_h_g.append(lstm_h)
                        if stochastic:
                            lstm_input_mu_var = tf.layers.conv2d(
                                lstm_h,
                                code_size * 2,
                                5,
                                strides=1,
                                padding = "same",
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=weight_decay),
                                name=f'convLSTM_mu{i}',
                                reuse=lstm_reuse)

                            lstm_input_mu, lstm_input_log_var = tf.split(lstm_input_mu_var, 2, axis=3)
                            g_mu_var.append((lstm_input_mu, lstm_input_log_var))
                            epsilon = tf.random_normal(tf.shape(lstm_input_mu))
                            lstm_z = lstm_input_mu + tf.exp(0.5 * lstm_input_log_var) * epsilon
                            lstm_input = tf.concat([lstm_input, lstm_z], 3)


                        lstm_input_all = tf.layers.conv2d(
                            lstm_input,
                            lstm_size * 4,
                            5,
                            strides=1,
                            padding = "same",
                            activation = None,
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=weight_decay),
                            name=f'convLSTM_input_c{i}',
                            reuse=lstm_reuse)
                        lstm_input_c, lstm_input_i1, lstm_input_i2, lstm_out = tf.split(lstm_input_all, 4, axis=3)
                        lstm_input_c = tf.nn.sigmoid(lstm_input_c)
                        lstm_input_i1 = tf.nn.sigmoid(lstm_input_i1)
                        lstm_input_i2 = tf.nn.tanh(lstm_input_i2)
                        lstm_out = tf.nn.sigmoid(lstm_out)

                        lstm_input = tf.multiply(lstm_input_i1, lstm_input_i2)
                        lstm_c = tf.multiply(lstm_c, lstm_input_c) + lstm_input
                        lstm_h = tf.multiply(tf.nn.tanh(lstm_c), lstm_out)

                        lstm_final_out = tf.layers.conv2d_transpose(
                            lstm_h,
                            n_filters,
                            4,
                            strides=(4, 4),
                            padding = "same",
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=weight_decay),
                            name=f'convLSTM_o_{i}',
                            reuse=lstm_reuse)

                        lstm_u = lstm_u + lstm_final_out
                    out = lstm_u

                    out = tf.layers.conv2d(
                            out,
                            n_filters,
                            1,
                            strides=(1, 1),
                            padding = "same",
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=weight_decay),
                            name=f'convLSTM_final_{i}',
                            reuse=reuse)

                    e_mu_var = []
                    if stochastic:
                        lstm_x_q_e = tf.layers.conv2d(
                            output_image,
                            16,
                            5,
                            strides=4,
                            padding = "same",
                            activation=tf.nn.sigmoid,
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=weight_decay),
                            name=f'convLSTM_x_q_e_{i}',
                            reuse=reuse)
                        lstm_h_e = tf.zeros([input_shape[0], int(h)//4, int(w)//4, code_size], dtype=tf.float32)
                        lstm_c_e = tf.zeros([input_shape[0], int(h)//4, int(w)//4, code_size], dtype=tf.float32)

                        for step_id in range(number_steps):

                            lstm_e_reuse = reuse or step_id > 0
                            lstm_input_mu_var_e = tf.layers.conv2d(
                                lstm_h_e,
                                code_size * 2,
                                5,
                                strides=1,
                                padding = "same",
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=weight_decay),
                                name=f'convLSTM_mu_e{i}',
                                reuse=lstm_e_reuse)
                            lstm_input_mu_e, lstm_input_log_var_e = tf.split(lstm_input_mu_var_e, 2, axis=3)

                            e_mu_var.append((lstm_input_mu_e, lstm_input_log_var_e))
                            if step_id == number_steps - 1:
                                break

                            lstm_u_e = tf.layers.conv2d(
                               lstm_u_g[step_id],
                               16,
                               5,
                               strides=4,
                               padding = "same",
                               activation=tf.nn.sigmoid,
                               kernel_initializer=tf.contrib.layers.xavier_initializer(),
                               kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=weight_decay),
                               name=f'convLSTM_u_e_{i}',
                               reuse=lstm_e_reuse)
                            lstm_input_e = tf.concat([lstm_h_e, lstm_h_g[step_id], lstm_u_e, lstm_r, lstm_v, lstm_x_q_e], axis=3)

                            lstm_input_all = tf.layers.conv2d(
                                lstm_input_e,
                                code_size * 4,
                                5,
                                strides=1,
                                padding = "same",
                                activation=None,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=weight_decay),
                                name=f'convLSTM_input_c_e{i}',
                                reuse=lstm_e_reuse)

                            lstm_input_c_e, lstm_input_i1_e, lstm_input_i2_e, lstm_out_e = tf.split(lstm_input_all, 4, axis=3)
                            lstm_input_c_e = tf.nn.sigmoid(lstm_input_c_e)
                            lstm_input_i1_e = tf.nn.sigmoid(lstm_input_i1_e)
                            lstm_input_i2_e = tf.nn.tanh(lstm_input_i2_e)
                            lstm_out_e = tf.nn.sigmoid(lstm_out_e)

                            lstm_input_e = tf.multiply(lstm_input_i1_e, lstm_input_i2_e)
                            lstm_c_e = tf.multiply(lstm_c_e, lstm_input_c_e) + lstm_input_e
                            lstm_h_e = tf.multiply(tf.nn.tanh(lstm_c_e), lstm_out_e)

                    kl_loss = 0
                    for layer_id, g_mu_var_ in enumerate(g_mu_var):
                        kl_loss += tf.reduce_mean(compute_kl_loss(
                            e_mu_var[layer_id][0],
                            e_mu_var[layer_id][1],
                            g_mu_var_[0],
                            g_mu_var_[1]
                        ))

                    if isinstance(kl_loss, int):
                        kl_loss = tf.constant(0.0, dtype = tf.float32)
                        
                    extra['kl_loss'] = kl_loss
            else:
                assert( 1 ==2, f"Unknown layer type: {layer_type}")


    return out, extra

