import tensorflow as tf


def warper(frame, flow, name="warper"):
    with tf.variable_scope(name):
        shape = flow.get_shape()
        bs, h, w, c = shape
        warp, occ = transformer(frame, flow, (int(h), int(w)))
        return warp, occ


def transformer(im, flow, out_size, name='SpatialTransformer', **kwargs):

    def _repeat(x, n_repeats):
        with tf.variable_scope('_repeat'):
            rep = tf.transpose(
                tf.expand_dims(tf.ones(shape=tf.stack([n_repeats, ])), 1), [1, 0])
            rep = tf.cast(rep, 'int32')
            x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
            return tf.reshape(x, [-1])

    def _interpolateHW(im, x, y, out_size):
        with tf.variable_scope('_interpolateHW'):
            # constants
            nBatch = tf.shape(im)[0]
            height = tf.shape(im)[1]
            width = tf.shape(im)[2]
            channels = tf.shape(im)[3]

            x = tf.cast(x, 'float32')
            y = tf.cast(y, 'float32')
            height_f = tf.cast(height, 'float32')
            width_f = tf.cast(width, 'float32')
            out_height = out_size[0]
            out_width = out_size[1]

            # clip coordinates to [0, dim-1]
            x = tf.clip_by_value(x, 0, width_f - 1)
            y = tf.clip_by_value(y, 0, height_f - 1)

            x0_f = tf.floor(x)
            y0_f = tf.floor(y)
            x1_f = x0_f + 1
            y1_f = y0_f + 1
            x0 = tf.cast(x0_f, 'int32')
            y0 = tf.cast(y0_f, 'int32')
            x1 = tf.cast(tf.minimum(x1_f, width_f - 1), 'int32')
            y1 = tf.cast(tf.minimum(y1_f, height_f - 1), 'int32')
            dim2 = width
            dim1 = width * height
            base = _repeat(tf.range(nBatch) * dim1, out_height * out_width)
            base_y0 = base + y0 * dim2
            base_y1 = base + y1 * dim2
            idx_a = base_y0 + x0
            idx_b = base_y1 + x0
            idx_c = base_y0 + x1
            idx_d = base_y1 + x1

            # use indices to lookup pixels in the flat image and restore channels dim
            im_flat = tf.reshape(im, tf.stack([-1, channels]))
            Ia = tf.gather(im_flat, idx_a)
            Ib = tf.gather(im_flat, idx_b)
            Ic = tf.gather(im_flat, idx_c)
            Id = tf.gather(im_flat, idx_d)

            # grab the occluded pixels this flow uncovers
            d_a = tf.sparse_to_dense(tf.cast(idx_a, 'int32'),
                                     [nBatch * height * width], 1,
                                     default_value=0, validate_indices=False, name="d_a")
            d_a = tf.reshape(tf.tile(tf.reshape(tf.cast(d_a, 'float32'),
                                                [nBatch, height, width, 1]),
                                     [1, 1, 1, channels]), [-1, channels])

            d_b = tf.sparse_to_dense(tf.cast(idx_b, 'int32'),
                                     [nBatch * height * width], 1,
                                     default_value=0, validate_indices=False, name="d_b")
            d_b = tf.reshape(tf.tile(tf.reshape(tf.cast(d_b, 'float32'),
                                                [nBatch, height, width, 1]),
                                     [1, 1, 1, channels]), [-1, channels])

            d_c = tf.sparse_to_dense(tf.cast(idx_c, 'int32'),
                                     [nBatch * height * width], 1,
                                     default_value=0, validate_indices=False, name="d_c")
            d_c = tf.reshape(tf.tile(tf.reshape(tf.cast(d_c, 'float32'),
                                                [nBatch, height, width, 1]),
                                     [1, 1, 1, channels]), [-1, channels])

            d_d = tf.sparse_to_dense(tf.cast(idx_d, 'int32'),
                                     [nBatch * height * width], 1,
                                     default_value=0, validate_indices=False, name="d_d")
            d_d = tf.reshape(tf.tile(tf.reshape(tf.cast(d_d, 'float32'),
                                                [nBatch, height, width, 1]),
                                     [1, 1, 1, channels]), [-1, channels])

            # and finally calculate interpolated values
            wa = tf.expand_dims(((x1_f - x) * (y1_f - y)), 1)
            wb = tf.expand_dims(((x1_f - x) * (y - y0_f)), 1)
            wc = tf.expand_dims(((x - x0_f) * (y1_f - y)), 1)
            wd = tf.expand_dims(((x - x0_f) * (y - y0_f)), 1)
            warp = tf.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id])
            occ = tf.add_n([wa * d_a, wb * d_b, wc * d_c, wd * d_d])
            return warp, occ

    def _meshgridHW(height, width):
        with tf.variable_scope('_meshgridHW'):
            # This should be equivalent to:
            #  x_t, y_t = np.meshgrid(np.linspace(0, width-1, width),
            #                         np.linspace(0, height-1, height))
            #  ones = np.ones(np.prod(x_t.shape))
            #  grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])
            x_t = tf.matmul(
                tf.ones(shape=tf.stack([height, 1])),
                tf.transpose(tf.expand_dims(tf.linspace(0.0, width - 1, width), 1), [1, 0])
            )

            y_t = tf.matmul(tf.expand_dims(tf.linspace(0.0, height - 1, height), 1),
                            tf.ones(shape=tf.stack([1, width])))

            x_t_flat = tf.reshape(x_t, (1, -1))
            y_t_flat = tf.reshape(y_t, (1, -1))

            ones = tf.ones_like(x_t_flat)
            grid = tf.concat(axis=0, values=[x_t_flat, y_t_flat, ones])
            return grid

    def _transform(flow, im, out_size):

        with tf.variable_scope('_transform'):
            nBatch = tf.shape(im)[0]
            height = tf.shape(im)[1]
            width = tf.shape(im)[2]
            nChannels = tf.shape(im)[3]

            # grid of (x_t, y_t, 1), eq (1) in ref [1]
            height_f = tf.cast(height, 'float32')
            width_f = tf.cast(width, 'float32')
            out_height = out_size[0]
            out_width = out_size[1]
            grid = _meshgridHW(out_height, out_width)

            [x_f, y_f] = tf.unstack(flow, axis=3)
            x_f_flat = tf.expand_dims(tf.reshape(x_f, (nBatch, -1)), 1)
            y_f_flat = tf.expand_dims(tf.reshape(y_f, (nBatch, -1)), 1)
            zeros = tf.zeros_like(x_f_flat)
            flowgrid = tf.concat(axis=1, values=[x_f_flat, y_f_flat, zeros])

            grid = tf.expand_dims(grid, 0)
            grid = tf.reshape(grid, [-1])
            grid = tf.tile(grid, tf.stack([nBatch]))
            grid = tf.reshape(grid, tf.stack([nBatch, 3, -1]), name="grid")

            grid = grid + flowgrid
            x_s = tf.slice(grid, [0, 0, 0], [-1, 1, -1])
            y_s = tf.slice(grid, [0, 1, 0], [-1, 1, -1])
            x_s_flat = tf.reshape(x_s, [-1])
            y_s_flat = tf.reshape(y_s, [-1])

            w, o = _interpolateHW(
                im, x_s_flat, y_s_flat,
                out_size)
            warp = tf.reshape(w, tf.stack([nBatch,
                                           out_height,
                                           out_width,
                                           nChannels]),
                              name="warp")
            occ = tf.reshape(o, tf.stack([nBatch,
                                          out_height,
                                          out_width,
                                          nChannels]),
                             name="occ")
            return warp, occ

    with tf.variable_scope(name):
        warp, occ = _transform(flow, im, out_size)
        return warp, occ


def depth_and_rot_to_flow(Z, r, distance=4.0):

    shape = Z.get_shape()
    bs = int(shape[0])
    h = int(shape[1])
    w = int(shape[2])
    Z = tf.reshape(Z, (bs, h, w))
    [grid_x1, grid_y1] = meshgrid2D(bs, h, w)

    # the starting pointcloud
    XYZ = Camera2World(grid_x1, grid_y1, Z)
    [X, Y, Z] = tf.split(axis=2, num_or_size_splits=3, value=XYZ, name="splitXYZ")
    Z -= distance  # important to subtract off the distance
    XYZ = tf.concat([X, Y, Z], axis=2)

    # now rotate pointcloud as necessary.
    XYZ_transpose = tf.transpose(XYZ, perm=[0, 2, 1])
    XYZ_mm = tf.matmul(r, XYZ_transpose)
    XYZ_rot = tf.transpose(XYZ_mm, perm=[0, 2, 1])
    #t_tiled = tf.tile(tf.expand_dims(t,axis=1),[1,h*w,1],name="t_tiled")
    #XYZ2 = tf.add(XYZ_rot,t_tiled,name="XYZ2")
    XYZ2 = XYZ_rot

    # project pointcloud
    [X2, Y2, Z2] = tf.split(axis=2, num_or_size_splits=3, value=XYZ2, name="splitXYZ")
    Z2 += distance  # now add the distance back on here
    x2y2_flat = World2Camera(X2, Y2, Z2)
    [x2_flat, y2_flat] = tf.split(axis=2, num_or_size_splits=2,
                                  value=x2y2_flat, name="splitxyz_flat")

    # subtract the new 2D locations from the old ones to get optical flow
    x1_flat = tf.reshape(grid_x1, [bs, -1, 1], name="x1")
    y1_flat = tf.reshape(grid_y1, [bs, -1, 1], name="y1")
    flow_flat = tf.concat(axis=2, values=[x2_flat - x1_flat, y2_flat - y1_flat], name="flow_flat")
    flow = tf.reshape(flow_flat, [bs, h, w, 2], name="flow")
    return flow


def reproject(view, mask, depth, rot):
    print('WARNING -- untested!')
    flow = depth_and_rot_to_flow(depth, rot)
    flow *= mask
    pred, occ = warper(view, flow)
    unocc = 1 - occ
    return pred, unocc


def reprojects(stuffs, mask, depth, rot, distance=4.0):
    #in the case of going from 2 to 1
    #we need rot 12
    #we estimate flow12 using depth 1
    #we warp using view2 and flow12

    #f12 = zrt2flow(z1, rt12)
    #i1e = warper(i2, f12)

    #to create f12 from z1, we make the pc for v1, then rotate it to v2, then project and subtract
    flow = depth_and_rot_to_flow(depth, rot, distance=distance)  # * mask

    warped = []
    for stuff in stuffs:
        pred, occ = warper(stuff, flow)
        warped.append(pred)
    return warped, occ
