import tensorflow as tf
import numpy as np


def scatter_add_tensor(ref, indices, updates, name=None):
    #from https://github.com/tensorflow/tensorflow/issues/2358
    """
    Adds sparse updates to a variable reference.

    This operation outputs ref after the update is done. This makes it easier to chain operations
    that need to use the reset value.

    Duplicate indices are handled correctly: if multiple indices reference the same location,
    their contributions add.

    Requires updates.shape = indices.shape + ref.shape[1:].
    :param ref: A Tensor. Must be one of the following types: float32, float64, int64, int32, uint8,
        uint16, int16, int8, complex64, complex128, qint8, quint8, qint32, half.

    :param indices: A Tensor. Must be one of the following types: int32, int64.
        A tensor of indices into the first dimension of ref.
    :param updates: A Tensor. Must have the same dtype as ref. A tensor of updated values to add
        to ref
    :param name: A name for the operation (optional).
    :return: Same as ref. Returned as a convenience for operations that want to use
        the updated values after the update is done.
    """
    with tf.name_scope(name, 'scatter_add_tensor', [ref, indices, updates]) as scope:
        ref = tf.convert_to_tensor(ref, name='ref')
        indices = tf.expand_dims(tf.convert_to_tensor(indices, name='indices'), axis=1)
        updates = tf.convert_to_tensor(updates, name='updates')
        ref_shape = tf.shape(ref, out_type=indices.dtype, name='ref_shape')
        scattered_updates = tf.scatter_nd(indices, updates, ref_shape, name='scattered_updates')
        with tf.control_dependencies(
                [tf.assert_equal(ref_shape, tf.shape(scattered_updates, out_type=indices.dtype))]
        ):
            output = tf.add(ref, scattered_updates, name=scope)
        return output


def bilinear_scatter(ref, indices, weights, quantity=1.0):
    assert len(indices) == len(weights) == 4

    indices = tf.concat(indices, axis=0)
    weights = tf.concat([weight * quantity for weight in weights], axis=0)
    ref = tf.scatter_add(ref, indices, weights)
    return ref


def render0(pts):
    X = pts[:, 0]
    Y = pts[:, 1]
    Z = pts[:, 2]
    C = pts[:, 3:6]
    A = pts[:, 6]
    x, y = World2Camera_(X, Y, Z)

    x0 = tf.floor(x)
    y0 = tf.floor(y)
    x1 = x0 + 1
    y1 = y0 + 1

    #"reverse" bilinear interpolation weights
    w_x0y0 = (x1 - x) * (y1 - y)
    w_x0y1 = (x1 - x) * (y - y0)
    w_x1y0 = (x - x0) * (y1 - y)
    w_x1y1 = (x - x0) * (y - y0)
    weights = [w_x0y0, w_x0y1, w_x1y0, w_x1y1]

    #weight closer points more to bring them to the front
    _min_z = tf.reduce_min(Z)
    _max_z = tf.reduce_max(Z)
    _z = (Z - _min_z) / (_max_z - _min_z + const.eps)
    depth_weight = tf.exp(-1.0 * _z)
    depth_and_alpha = depth_weight * A

    weights_da = [tf.expand_dims(depth_and_alpha * weight, axis=1) for weight in weights]
    weights_ = [tf.expand_dims(weight, axis=1) for weight in weights]

    x0 = tf.cast(x0, tf.int32)
    y0 = tf.cast(y0, tf.int32)
    x1 = tf.cast(x1, tf.int32)
    y1 = tf.cast(y1, tf.int32)

    idx_x0y0 = y0 * const.W + x0
    idx_x1y0 = y0 * const.W + x1
    idx_x0y1 = y1 * const.W + x0
    idx_x1y1 = y1 * const.W + x1

    #create 'ref' img and update it with scatter
    img = tf.Variable(0.0)
    norm = tf.Variable(0.0)
    depth = tf.Variable(0.0)
    alpha = tf.Variable(0.0)
    alpha_norm = tf.Variable(0.0)

    img_zero = tf.zeros((const.H * const.W, 3))
    ch_zero = tf.zeros((const.H * const.W, 1))

    #need to zero out variables every iteration
    img = tf.assign(img, img_zero, validate_shape=False)
    norm = tf.assign(norm, ch_zero, validate_shape=False)
    depth = tf.assign(depth, ch_zero, validate_shape=False)
    alpha = tf.assign(alpha, ch_zero, validate_shape=False)
    alpha_norm = tf.assign(alpha_norm, ch_zero, validate_shape=False)

    indices = [idx_x0y0, idx_x0y1, idx_x1y0, idx_x1y1]

    img = bilinear_scatter(img, indices, weights_da, C)
    depth = bilinear_scatter(depth, indices, weights_da, tf.expand_dims(Z, 1))
    norm = bilinear_scatter(norm, indices, weights_da)

    alpha = bilinear_scatter(alpha, indices, weights_, tf.expand_dims(A, 1))
    alpha_norm = bilinear_scatter(alpha_norm, indices, weights_)

    #in the case of 'collisions', we average the colors by the weights
    img = img / (norm + const.eps)
    depth = depth / (norm + const.eps)
    alpha = alpha / (alpha_norm + const.eps)

    img = tf.reshape(img, (const.H, const.W, 3))
    depth = tf.reshape(depth, (const.H, const.W, 1))
    alpha = tf.reshape(alpha, (const.H, const.W, 1))
    vis = tf.cast(tf.reshape(norm > 0.0, (const.H, const.W, 1)), tf.float32)

    return img, depth, alpha, vis


def render1(pts, nb=5):
    bin_idx = bin_indices(pts[:, 2], nb)
    binned_pts = tf.dynamic_partition(pts, bin_idx, nb)[::-1]  # reverse!

    img = tf.zeros((const.H, const.W, 3))
    depth = tf.zeros((const.H, const.W, 1))
    alpha = tf.zeros((const.H, const.W, 1))
    vis = tf.zeros((const.H, const.W, 1))
    for pts_ in binned_pts:

        next_img, next_depth, next_alpha, next_vis = render0(pts_)
        img = img * (1 - next_alpha) + next_img * next_alpha
        depth = depth * (1 - next_alpha) + next_depth * next_alpha
        alpha = tf.maximum(alpha, next_alpha)
        vis = tf.maximum(vis, next_vis)

    img *= vis
    img += (1.0 - vis)
    depth *= vis
    depth += (1.0 - vis) * 4.0

    return img, depth, alpha, vis


def render_pts_from_pose(pts, theta, phi, do_preprocess=False):
    if len(pts.get_shape()) == 3:
        assert const.BS == 1
        pts = pts[0]

    if do_preprocess:
        pts_ = preprocess_threshold_sparsity(pts)
    else:
        pts_ = pts

    coords = pts_[:, :3]
    tail = pts_[:, 3:]

    rot_mat = tf_rotate_matrix(-theta, phi)
    coords = tf.transpose(tf.matmul(rot_mat, tf.transpose(coords)))
    coords = coords + tf.constant([0.0, 0.0, 4.0])

    pts_ = tf.concat([coords, tail], axis=1)

    with tf.name_scope('render'):
        out = render1(pts_)

    return out


def reproject_all(views, depths, masks):
    with tf.name_scope('reproject_lift'):
        pcs = [Z_to_PC(depth) for depth in tf.unstack(depths, axis=3)]
        pcs = [normalize_pc(pc, i) for i, pc in enumerate(pcs)]
        f = lambda x: tf.reshape(x, (const.BS, const.H * const.W, -1))
        views = list(map(f, tf.unstack(views, axis=3)))
        masks = list(map(f, tf.unstack(masks, axis=3)))

        pts = []
        for (pc, view, mask) in zip(pcs, views, masks):
            pt = tf.concat([pc, view, mask], axis=2)  # bs, ?, 7
            pts.append(pt)

        #now we have all pointclouds in normalized form
        exclude_pts = [pts[:i] + pts[i + 1:] for i in range(const.V)]
        exclude_pts = [tf.concat(x, axis=1) for x in exclude_pts]
        phis = [const.PHI_IDX * 10.0 for i in range(const.V)]
        thetas = [i * 360.0 / const.V for i in range(const.V)]

    with tf.name_scope('reproject_sink'):
        outputs = [render_pts_from_pose(pts, theta, phi)
                   for (pts, theta, phi) in zip(exclude_pts, thetas, phis)]
        reprojs, depths, alphas, viss = list(zip(*outputs))

        reprojs = tf.concat(reprojs, axis=2)
        depths = tf.concat(depths, axis=2)
        alphas = tf.concat(alphas, axis=2)
        viss = tf.concat(viss, axis=2)

    #add the batching back
    reprojs = tf.expand_dims(reprojs, axis=0)
    depths = tf.expand_dims(depths, axis=0)
    alphas = tf.expand_dims(alphas, axis=0)
    viss = tf.expand_dims(viss, axis=0)

    return reprojs, depths, alphas, viss
