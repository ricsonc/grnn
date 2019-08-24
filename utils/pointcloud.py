import tensorflow as tf
import constants as const
import numpy as np

from . import tfutil
from . import camera


def Z_to_PC(Z):
    #BS x H x W x 1 -> BS x H*W x 3
    shape = Z.get_shape()
    bs = int(shape[0])
    h = int(shape[1])
    w = int(shape[2])
    Z = tf.reshape(Z, (bs, h, w))
    [grid_x1, grid_y1] = tfutil.meshgrid2D(bs, h, w)
    XYZ = camera.Camera2World(grid_x1, grid_y1, Z)
    return XYZ


def Z_to_PC_dxdy(Z, dx, dy):
    #BS x H x W x 1 -> BS x H*W x 3
    shape = Z.get_shape()
    bs = int(shape[0])
    h = int(shape[1])
    w = int(shape[2])
    Z = tf.reshape(Z, (bs, h, w))
    [grid_x1, grid_y1] = tfutil.meshgrid2D(bs, h, w)
    if len(dx.get_shape()) == 4:
        dx = tf.squeeze(dx, axis=3)
        dy = tf.squeeze(dy, axis=3)
    grid_x1 += dx
    grid_y1 += dy
    XYZ = camera.Camera2World(grid_x1, grid_y1, Z)
    return XYZ


def normalize_pc(pc, i):
    #remove offset
    X, Y, Z = tf.split(pc, 3, axis=2)
    Z -= 4.0
    pc = tf.concat([X, Y, Z], axis=2)

    r = camera.rotate_matrix(i * 20, -10 * const.PHI_IDX, order='phi').astype(np.float32)
    r = tf.expand_dims(r, axis=0)
    pc_t = tf.transpose(pc, (0, 2, 1))
    pc = tf.transpose(tf.matmul(r, pc_t), (0, 2, 1))
    return pc


def Zs_to_PC(Zs, dxs=None, dys=None):
    #in: BS x H x W x V x 1
    #out: BS x H*W*V x 3
    Zs = tf.unstack(Zs, axis=3)
    if dxs is not None:
        dxs_ = tf.unstack(dxs, axis=3)
        dys_ = tf.unstack(dys, axis=3)
    else:
        dxs_ = list(range(len(Zs)))
        dys_ = list(range(len(Zs)))

    pcs = []
    for i, (Z, dx, dy) in enumerate(zip(Zs, dxs_, dys_)):
        #BS x H x W x 1

        if dxs is None and dys is None:
            pc = Z_to_PC(Z)
        else:
            assert (dxs is not None) and (dys is not None)
            pc = Z_to_PC_dxdy(Z, dx, dy)

        #remove offset
        X, Y, Z = tf.split(pc, 3, axis=2)
        Z -= 4.0
        pc = tf.concat([X, Y, Z], axis=2)

        r = camera.rotate_matrix(i * 20, -10 * const.PHI_IDX, order='phi').astype(np.float32)
        r = tf.expand_dims(r, axis=0)
        pc_t = tf.transpose(pc, (0, 2, 1))
        pc = tf.transpose(tf.matmul(r, pc_t), (0, 2, 1))

        pcs.append(pc)
    return tf.concat(pcs, axis=1)


def RGBAZs_to_CPC(RGBs, Zs, As, dx=None, dy=None):
    #in: BS x H x W x V x 1, BS x H x W x V x 3, BS x H x W x V x 1
    #out: BS x H*W*V x (3+3+1)
    PC = Zs_to_PC(Zs, dx, dy)
    f = lambda x, d: tf.reshape(tf.transpose(x, (0, 3, 1, 2, 4)), (const.BS, -1, d))
    RGBs = f(RGBs, 3)
    As = f(As, 1)
    CPC = tf.concat([PC, RGBs, As], axis=2)
    return CPC


def normalize_point_coords(pts, theta, phi):
    coords = pts[:, :3]
    tail = pts[:, 3:]

    rot_mat = camera.tf_rotate_matrix(-theta, phi)
    coords = tf.transpose(tf.matmul(rot_mat, tf.transpose(coords)))
    coords = coords + tf.constant([0.0, 0.0, 4.0])

    pts = tf.concat([coords, tail], axis=1)
    return pts


def preprocess_threshold_sparsity(pts):
    nb = 10000
    sparsity = 0.1

    alpha = pts[:, 6]

    #_, keep_top = tf.nn.top_k(alpha, k=int(nb/sparsity), sorted=False)
    #pts = tf.gather(pts, keep_top)
    #alpha = pts[:,6]

    threshold = tf.contrib.distributions.percentile(alpha, 50.0)

    keep_nz = tf.squeeze(tf.where(alpha >= threshold), axis=1)
    pts = tf.gather(pts, keep_nz)
    alpha = pts[:, 6]

    n = tf.cast(tf.shape(alpha)[0], tf.float32)
    sparsity = nb / n

    keep_rnd = tf.squeeze(tf.where(tf.random_uniform(tf.shape(alpha)) < sparsity), axis=1)
    pts = tf.gather(pts, keep_rnd)

    return pts
