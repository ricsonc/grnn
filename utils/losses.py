import tensorflow as tf
import constants as const
from . import embnet

from ipdb import set_trace as st


def l1loss_stackedmask(x, y, mask):
    f = lambda q: tf.reshape(q, [const.BS, const.H, const.W, const.V, -1])
    return l1loss(f(x), f(y), f(mask))


def l1loss(x, y, mask=None, verify_shape=True, stopgrad = True):
    if verify_shape:
        assert len(x.get_shape()) == len(y.get_shape())
    if mask is None:
        return tf.reduce_mean(tf.abs(x - y))
    else:
        mask = tf.stop_gradient(mask)
        # where the mask is 0, the loss is 0
        mask = tf.stop_gradient(mask)
        return tf.reduce_mean(tf.abs(mask * (x - y))) / (tf.reduce_mean(mask) + const.eps)


def l2loss(x, y, mask=None, verify_shape = True, strict = False):
    if verify_shape:
        assert len(x.get_shape()) == len(y.get_shape())
    if strict:
        xshp = x.shape.as_list()
        yshp = y.shape.as_list()
        assert (None not in xshp) and (None not in yshp) and xshp == yshp
    if mask is None:
        return tf.reduce_mean(tf.square(x - y))
    else:
        # where the mask is 0, the loss is 0
        return tf.reduce_mean(tf.square(mask * (x - y))) / (tf.reduce_mean(mask) + const.eps)


def binary_uncertainty(x, p=0.1):
    return x * (1 - p) + p / 2.0


def binary_ce_loss(x, y, mask=None, positive_weight = 1.0, negative_weight = 1.0):
    inner = (positive_weight * y * tf.log(x + const.eps) +
             negative_weight * (1 - y) * tf.log(1 - x + const.eps))
    
    if mask is None:
        return -tf.reduce_mean(inner)
    else:
        return -tf.reduce_mean(mask * inner) / (tf.reduce_mean(mask) + const.eps)


def two_way_ce_loss(x, y, mask=None, p=0.1):
    x = binary_uncertainty(x, p)
    y = binary_uncertainty(y, p)
    return binary_ce_loss(x, y, mask) + binary_ce_loss(y, x, mask)


def smoothLoss(x):
    bs, h, w, c = x.get_shape()
    kernel = tf.transpose(tf.constant([[[[0, 0, 0], [0, 1, -1], [0, 0, 0]]],
                                       [[[0, 0, 0], [0, 1, 0], [0, -1, 0]]]],
                                      dtype=tf.float32), perm=[3, 2, 1, 0],
                          name="kernel")
    diffs = [tf.nn.conv2d(tf.expand_dims(x_, axis=3), kernel, [1, 1, 1, 1],
                          padding="SAME", name="diff") for x_ in tf.unstack(x, axis=-1)]
    diff = tf.concat(axis=3, values=diffs)
    mask = tf.ones([bs, h - 1, w - 1, 1], name="mask")
    mask = tf.concat(axis=1, values=[mask, tf.zeros([bs, 1, w - 1, 1])])
    mask = tf.concat(axis=2, values=[mask, tf.zeros([bs, h, 1, 1])])
    loss = tf.reduce_mean(tf.abs(diff * mask), name="loss")
    return loss

def embedding_loss(x, y):
    #input: 2 of BxHxWxC
    #output: scalar
    #return tf.constant(0.0, dtype = tf.float32)

    rgb = tf.ones((const.BS, const.H, const.W, 3), dtype = tf.float32)
    return embnet.metric_loss(
        rgb,
        x, y,
        None,
        rgb[:,:,:,:1],
    )
