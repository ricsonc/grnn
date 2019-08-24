import tensorflow as tf
import constants as const
import numpy as np
import math
from . import tfutil


def make_batch_rot(r):
    return tf.tile(tf.reshape(tf.constant(r), (1, 3, 3)), (const.BS, 1, 1))


def tf_make_batch_rot(r):
    return tf.tile(tf.reshape(r, (1, 3, 3)), (const.BS, 1, 1))

#this is for numpy -- untested


def homogenize_transform(rotmat):
    if len(rotmat.shape) == 2:
        new_rotmat = np.zeros((4, 4))
        new_rotmat[:3, :3] = rotmat
        new_rotmat[3, 3] = 1.0
    elif len(roatmat.shape) == 3:
        bs = len(rotmat)
        new_rotmat = np.zeros((bs, 4, 4))
        new_rotmat[:, :3, :3] = rotmat
        new_rotmat[:, 3, 3] = 1.0
    else:
        raise Exception('bad rank')
    return new_rotmat


def tf_homogenize_transform(rotmat):
    assert tfutil.rank(rotmat) == 2

    right = tf.zeros((3, 1))
    bottom = tf.reshape(tf.constant([0.0, 0.0, 0.0, 1.0], dtype=tf.float32), (1, 4))
    rotmat = tf.concat([rotmat, right], axis=1)
    rotmat = tf.concat([rotmat, bottom], axis=0)
    return rotmat


def Camera2World(x, y, Z):
    f = lambda q: tf.tile(tf.reshape(tf.constant(q), (1, 1, 1)), [const.BS, const.H, const.W])
    fy = f(const.fy)
    fx = f(const.fx)
    x0 = f(const.x0)
    y0 = f(const.y0)

    X = (Z / fx) * (x - x0)
    Y = (Z / fy) * (y - y0)
    pointcloud = tf.stack([tf.reshape(X, [const.BS, -1]),
                           tf.reshape(Y, [const.BS, -1]),
                           tf.reshape(Z, [const.BS, -1])],
                          axis=2, name="world_pointcloud")
    return pointcloud


def World2Camera(X, Y, Z):
    f = lambda q: tf.tile(tf.reshape(tf.constant(q), (1, 1, 1)), [const.BS, const.H * const.W, 1])
    fy = f(const.fy)
    fx = f(const.fx)
    x0 = f(const.x0)
    y0 = f(const.y0)

    x = (X * fx) / (Z + const.eps) + x0
    y = (Y * fy) / (Z + const.eps) + y0

    proj = tf.concat(axis=2, values=[x, y], name="camera_projection")
    return proj


def World2Camera_arbitraryshape(X, Y, Z):
    #this one does not require pts shaped like an img
    fy = const.fy
    fx = const.fx
    x0 = const.x0
    y0 = const.y0

    x = (X * fx) / (Z + const.eps) + x0
    y = (Y * fy) / (Z + const.eps) + y0

    return x, y


def rotate_matrix_at_elev(theta, phi):
    r1 = rotate_matrix(0.0, phi)
    r2 = rotate_matrix(theta, 0.0)
    r3 = rotate_matrix(0.0, -phi)
    #order of composition matters
    return np.matmul(r3, np.matmul(r2, r1))


def tf_rotate_matrix_at_elev(theta, phi):
    r1 = tf_rotate_matrix(0.0, -phi)
    r2 = tf_rotate_matrix(theta, 0.0)
    r3 = tf_rotate_matrix(0.0, phi)
    #order of composition matters
    return tf.matmul(r3, tf.matmul(r2, r1))


def rotate_matrix(theta, phi=0.0, order=''):
    t = theta * math.pi / 180.0
    p = phi * math.pi / 180.0

    cp = math.cos(p)
    sp = math.sin(p)
    ct = math.cos(t)
    st = math.sin(t)

    rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0, cp, -sp],
        [0.0, sp, cp]
    ])
    ry = np.array([
        [ct, 0.0, st],
        [0.0, 1.0, 0.0],
        [-st, 0.0, ct]
    ])

    if order == 'phi':
        rval = np.matmul(ry, rx)
    else:
        rval = np.matmul(rx, ry)
    return rval


def tf_rotate_matrix(theta, phi=0.0, order=''):
    t = theta * (math.pi / 180.0)
    p = phi * (math.pi / 180.0)

    cp = tf.cos(p)
    sp = tf.sin(p)
    ct = tf.cos(t)
    st = tf.sin(t)

    rx = tf.stack([
        tf.stack([1.0, 0.0, 0.0]),
        tf.stack([0.0, cp, -sp]),
        tf.stack([0.0, sp, cp])
    ])
    ry = tf.stack([
        tf.stack([ct, 0.0, st]),
        tf.stack([0.0, 1.0, 0.0]),
        tf.stack([-st, 0.0, ct])
    ])

    if order == 'phi':
        rval = tf.matmul(ry, rx)
    else:
        rval = tf.matmul(rx, ry)
    return rval
