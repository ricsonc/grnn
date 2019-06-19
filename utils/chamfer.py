import tensorflow as tf
from . import camera
from . import tfutil
import constants as const
from . import camera


def normalized_visibility(coords, nb=10):
    ''' input: normalized 3d coords, output: visibility indices '''

    bin_idx = tfutil.bin_indices(coords[:, 2], nb)
    binned_coords = tf.dynamic_partition(coords, bin_idx, nb)[::-1]  # reverse to go back to front

    n = tf.shape(coords)[0]
    idxs = tf.range(1, n + 1)  # to avoid 0
    binned_idxs = tf.dynamic_partition(idxs, bin_idx, nb)[::-1]

    idx_map = tf.Variable(0, dtype=tf.int32)
    idx_zero = tf.zeros((const.H * const.W,), dtype=tf.int32)
    idx_map = tf.assign(idx_map, idx_zero, validate_shape=False)

    for coords, idxs in zip(binned_coords, binned_idxs):
        X = coords[:, 0]
        Y = coords[:, 1]
        Z = coords[:, 2]
        x, y = camera.World2Camera_arbitraryshape(X, Y, Z)

        x_ = tfutil.prob_round(x)
        y_ = tfutil.prob_round(y)

        idx = y_ * const.W + x_  # these are scatter destination indices, not point cloud indices

        #if anything is out of range, then don't scatter it...
        valid = tf.logical_and(tf.greater(idx, -1), tf.less(idx, const.H * const.W))
        idx = tf.boolean_mask(idx, valid)
        vals_to_scatter = tf.boolean_mask(idxs, valid)

        idx_map = tf.scatter_update(idx_map, idx, vals_to_scatter)

    vis_idx = tf.boolean_mask(idx_map, idx_map > 0) - 1  # subtract off the offset

    return vis_idx, idx_map


def pc_visibility(pts, theta, phi):
    ''' inputs: points and a pose, output: visibility indices'''
    if len(pts.get_shape()) == 3:
        assert const.BS == 1
        pts = pts[0]

    coords = pts[:, :3]
    rot_mat = camera.tf_rotate_matrix(-theta, phi)
    coords = tf.transpose(tf.matmul(rot_mat, tf.transpose(coords)))
    coords = coords + tf.constant([0.0, 0.0, 4.0])

    with tf.name_scope('render'):
        vis_idx, idx_map = normalized_visibility(coords)

    return vis_idx, idx_map


def chamfer_preprocess(pts):
    ''' trims off points with mask < 0.1 and samples the rest according to the confidence'''
    if len(pts.get_shape()) == 3:
        pts = pts[0]

    alpha = pts[:, 6]
    keep_nz = tf.squeeze(tf.where(alpha >= 0.1), axis=1)
    pts = tf.gather(pts, keep_nz)

    alpha = pts[:, 6]
    keep_rnd = tf.squeeze(tf.where(tf.random_uniform(tf.shape(alpha)) < alpha), axis=1)
    pts = tf.gather(pts, keep_rnd)
    return pts

def batch_dist_mat(bpts1, bpts2):
    return tf.map_fn(lambda x: dist_mat(*x), [bpts1, bpts2], dtype = tf.float32, parallel_iterations = const.BS)

def dist_mat(pts1, pts2):
    #(x-y)^2 = x^2 + y^2 - 2xy
    xy = tf.matmul(pts1, tf.transpose(pts2))  # n by m
    xsq = tf.expand_dims(tf.reduce_sum(tf.square(pts1), axis=1), axis=1)  # n by 1
    ysq = tf.expand_dims(tf.reduce_sum(tf.square(pts2), axis=1), axis=0)  # 1 by m
    return xsq + ysq - 2 * xy


def min_dist(pts1, pts2):
    ''' for each pt in pts2, return index of closest point in pts1 '''
    distmat = dist_mat(pts1, pts2)
    return tf.argmin(distmat, axis=0)


def render(pts, idx_map):
    assert const.BS == 1
    #can only render view and mask correctly
    flatmask = 1.0 - tf.cast(tf.equal(idx_map, 0), tf.float32)  # 0 where idx_map == 0
    flatimg = tf.gather(pts, idx_map - 1)  # idx_map is off by 1!
    flatview = flatimg[:, 3:6]
    flatdepth = flatimg[:, 2]
    flatmask *= flatimg[:, 6]  # multiply by gather mask

    reshape = lambda x: tf.reshape(x, (const.H, const.W, -1))
    view = reshape(flatview)
    mask = reshape(flatmask)
    depth = reshape(flatdepth)

    view = view * mask + (1.0 - mask)
    depth = depth * mask + (1.0 - mask)

    #memory debug
    #import numpy as np
    #view = tf.constant(np.zeros(view.get_shape(), dtype = np.float32))
    #depth = tf.constant(np.zeros(depth.get_shape(), dtype = np.float32))
    #mask = tf.constant(np.zeros(mask.get_shape(), dtype = np.float32))
    return view, depth, mask
