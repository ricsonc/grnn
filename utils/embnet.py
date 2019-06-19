# from scipy.misc import imsave
# import cv2
import os, itertools
import tensorflow as tf
# from utils_basic import *
# from decode_3D_to_2D import *
# import utils_improc
# import utils_geom
# import utils_misc
from ipdb import set_trace as st

from munch import Munch

hyp = Munch(
    emb_samp = 'gridrand',
    emb_use_aug = False,
    emb_grid_cell_sz = [8, 8],
    emb_do_subsamp = True,
)


import tensorflow.contrib.slim as slim
import numpy as np
from sklearn.decomposition import PCA
tfml = tf.contrib.losses.metric_learning

EPS = 1e-4

def normalize(d):
    # d is B x whatever. normalize within each element of the batch
    return tf.map_fn(normalize_single, (d), dtype=tf.float32)

def normalize_single(d):
    dmin = tf.reduce_min(d)
    dmax = tf.reduce_max(d)
    d = (d-dmin)/(EPS+(dmax-dmin))
    return d

def batch_norm(x, istrain):
    # return tf.identity(x)
    # decay of 0.99 can take ~1k steps to learn (according to my plots)
    return tf.contrib.layers.batch_norm(x, decay=0.9, 
                                        is_training=istrain,
                                        # updates_collections=None,
                                        center=True,
                                        scale=True,
                                        reuse=False)

def gradient(x, absolute=False, square=False):
    # x should be B x H x W x C
    dy = x[:, 1:, :, :] - x[:, :-1, :, :]
    dx = x[:, :, 1:, :] - x[:, :, :-1, :]
    zeros = tf.zeros_like(x)
    zero_row = tf.expand_dims(zeros[:, 0, :, :], axis=1)
    zero_col = tf.expand_dims(zeros[:, :, 0, :], axis=2)
    dy = tf.concat([dy, zero_row], axis=1)
    dx = tf.concat([dx, zero_col], axis=2)
    if absolute:
        dx = tf.abs(dx)
        dy = tf.abs(dy)
    if square:
        dx = tf.square(dx)
        dy = tf.square(dy)
    return dx, dy

def edge_aware_smooth_loss(x, rgb):
    S, H, W, C = x.shape.as_list()
    # x should be B x H x W x C
    # rgb should be B x H x W x 3
    # penalizes smoothness except when the RGB is not smooth
    X_dx, X_dy = gradient(x)
    R_dx, R_dy = gradient(rgb)
    
    # eq 3 in https://arxiv.org/pdf/1609.03677.pdf 
    X_dx, X_dy = gradient(x)
    R_dx, R_dy = gradient(rgb)
    X_dx = tf.reduce_sum(tf.abs(X_dx), axis=3, keepdims=True)
    X_dy = tf.reduce_sum(tf.abs(X_dy), axis=3, keepdims=True)
    R_dx = tf.exp(-tf.reduce_sum(tf.abs(R_dx), axis=3, keepdims=True))
    R_dy = tf.exp(-tf.reduce_sum(tf.abs(R_dy), axis=3, keepdims=True))
    l_x = X_dx*R_dx
    l_y = X_dy*R_dy
    s = l_x + l_y
    s = tf.reduce_mean(s)
    return s

def texturedness(rgb, scale=1.0):
    # rgb should be B x H x W x 3
    # shows where the RGB is not smooth
    if not scale==1.0:
        h = tf.cast(H*scale, tf.int32)
        w = tf.cast(W*scale, tf.int32)
    R_dx, R_dy = gradient(rgb)
    R_dx = 1.0-tf.exp(-tf.reduce_sum(tf.abs(R_dx), axis=3, keepdims=True))
    R_dy = 1.0-tf.exp(-tf.reduce_sum(tf.abs(R_dy), axis=3, keepdims=True))
    t = R_dx+R_dy
    if not scale==1.0:
        t = tf.image.resize_images(t, [H, W])
    return t

def get_textured_inds_single(tex):
    ## tex -- [H/2,W/2]
    ## Helper for get_textured_inds.
    ## Computes samples based on texture.
    ind = tf.cast(tf.where(tf.greater(tex, 0.2)), tf.int32)
    ind = tf.random_shuffle(ind)
    if hyp.emb_samp is 'tex':
        ## Pure texture sampling. 
        ESS = hyp.emb_samp_sz
        ind = ind[:ESS]
    else:
        ## Sample texture inds in grid to ensure one sample per cell.
        ## If not texture sample is available for a cell, use rand sample.
        H,W = tex.shape.as_list()
        GH,GW = hyp.emb_grid_cell_sz
        ESSH,ESSW = H//GH,W//GW
        ESS = ESSH*ESSW
        ## Generate rand inds in case its needed.
        ## Generate random row,col inds for every cell.
        indr = tf.random.uniform([ESS,1], maxval=GH, dtype=tf.int32)
        indc = tf.random.uniform([ESS,1], maxval=GW, dtype=tf.int32)
        ## Add cell corner as offset to obtain distinct cells.
        indrc = list(itertools.product(list(range(0,H,GH)),list(range(0,W,GW))))
        indrc = tf.constant(indrc, shape=[ESS,2], dtype=tf.int32)
        indrc += tf.concat([indr,indc], axis=1)
        ## Concat texture pts with gridrand pts to ensure at-least one pt
        ## per cell.
        ind = tf.concat([ind, indrc], axis=0)
        ind_g = ind//tf.constant([GH,GW], shape=[1,2], dtype=tf.int32)
        ind_ = list()
        ## Get sample for every cell.
        for r in range(ESSH):
            for c in range(ESSW):
                i = tf.logical_and(tf.equal(ind_g[:,0],r),
                                   tf.equal(ind_g[:,1],c))
                i = tf.where(i)
                ind_.append(tf.reshape(ind[i[0,0]], [1,2]))
        ind = tf.concat(ind_, axis=0)
        if hyp.emb_do_subsamp:
            ## Randomly choose even/odd columns in the grid and retain only
            ## those pts. 4x4 grid has 16 pts. We retain 2x2 giving 4 pts.
            beg = tf.random.uniform([2], minval=0, maxval=2, dtype=tf.int32)
            rows,cols = tf.meshgrid(tf.range(beg[0],ESSH,2), tf.range(beg[1],ESSW,2))
            ESS_SUB = (ESSH//2)*(ESSW//2)
            rows,cols = tf.reshape(rows,[ESS_SUB,1]), tf.reshape(cols,[ESS_SUB,1])
            rowscols = tf.concat([rows,cols], axis=1)
            ind = tf.reshape(ind, [ESSH,ESSW,2])
            ind = tf.gather_nd(ind, rowscols)
    return ind

def get_textured_inds(rgb):
    ## rgb -- [S,H,W,C]
    ## Sample inds based on texture.
    S,H,W,C = rgb.shape.as_list()
    tex = texturedness(rgb)
    tex = normalize(tex)
    tex = tf.squeeze(tex, axis=3)
    ind = tf.map_fn(get_textured_inds_single, (tex),
                    dtype=tf.int32)  # 
    return ind

def indAssign(rgb, ind):
    rgb[ind[:,0],ind[:,1]] = 0.5
    return rgb

def getSampleInds(rgb):
    ## rgb -- [S,H,W,C]
    ## Get indices [ESS,2] to use for pixelwise loss.
    S,H,W,C = rgb.shape.as_list()

    if hyp.emb_samp in 'tex':
        ## Texture only.
        ESS = hyp.emb_samp_sz
        ind = get_textured_inds(rgb)
    elif hyp.emb_samp is 'gridtex':
        ## Sample texture inds from a grid. One sample per cell.
        GH,GW = hyp.emb_grid_cell_sz
        ESSH,ESSW = H//GH,W//GW
        ESS = ESSH*ESSW
        ind = get_textured_inds(rgb)
    elif hyp.emb_samp is 'rand':
        ## Rand sampling of inds.

        ESS = hyp.emb_samp_sz
        indr = tf.random.uniform([S,ESS,1], maxval=H, dtype=tf.int32)
        indc = tf.random.uniform([S,ESS,1], maxval=W, dtype=tf.int32)
        ind = tf.concat([indr,indc], axis=2)
    elif hyp.emb_samp is 'gridrand':
        ## Sample random inds from a grid. One sample per cell.
        GH,GW = hyp.emb_grid_cell_sz
        ESSH,ESSW = H//GH,W//GW
        ESS = ESSH*ESSW
        ind = get_textured_inds(tf.zeros_like(rgb))
    return ind,ESS

def pca_embed(emb, keep):
    ## emb -- [S,H/2,W/2,C]
    ## keep is the number of principal components to keep
    ## Helper function for reduce_emb.
    emb_reduced = list()
    for img in emb:
        H, W, K = np.shape(img)
        if np.isnan(img).any():
            out_img = np.zeros([H,W,keep], dtype=img.dtype)
            emb_reduced.append(out_img)
            continue
        pixelskd = np.reshape(img, (H*W, K))
        P = PCA(keep)
        P.fit(pixelskd)
        pixels3d = P.transform(pixelskd)
        out_img = np.reshape(pixels3d, [H,W,keep])
        emb_reduced.append(out_img)
    emb_reduced = np.stack(emb_reduced, axis=0).astype(np.float32)
    return emb_reduced

def pca_embed_together(emb, keep):
    ## emb -- [S,H/2,W/2,C]
    ## keep is the number of principal components to keep
    ## Helper function for reduce_emb.
    S, H, W, K = np.shape(emb)
    if np.isnan(emb).any():
        out_img = np.zeros([S,H,W,keep], dtype=img.dtype)
    pixelskd = np.reshape(emb, (S*H*W, K))
    P = PCA(keep)
    P.fit(pixelskd)
    pixels3d = P.transform(pixelskd)
    out_img = np.reshape(pixels3d, [S,H,W,keep]).astype(np.float32)
    return out_img

def reduce_emb(emb, inbound=None, together=False):
    ## emb -- [S,H/2,W/2,C], inbound -- [S,H/2,W/2,1]
    ## Reduce number of chans to 3 with PCA. For vis.
    S,H,W,C = emb.shape.as_list()
    keep = 3
    if together:
        emb = tf.py_func(pca_embed_together, [emb,keep], tf.float32)
    else:
        emb = tf.py_func(pca_embed, [emb,keep], tf.float32)
    emb.set_shape([S,H,W,keep])
    emb = normalize(emb) - 0.5
    if inbound is not None:
        emb_inbound = emb*inbound
    else:
        emb_inbound = None
    return emb, emb_inbound

def emb_vis(rgb, emb, emb_pred, inbound):
    ## emb,emb_pred -- [S,H/2,W/2,C] where C is length of emb vector per pixel.
    ## rgb -- [S,H/2,W/2,3], inbound -- [S,H/2,W/2,1]
    S,H,W,C = emb.shape.as_list()
    embs = tf.concat([emb, emb_pred], axis=0)
    inbounds = tf.concat([inbound, inbound], axis=0)
    # emb, emb_inbound = reduce_emb(emb, inbound)
    # emb_pred, emb_pred_inbound = reduce_emb(emb_pred, inbound)
    
    _, embs_inbound = reduce_emb(embs, inbounds, together=True)
    emb_inbound, emb_pred_inbound = tf.split(embs_inbound, 2, axis=0)
    
    rgb_emb_vis = tf.concat([rgb, emb_inbound, emb_pred_inbound], axis=2)
    
    print('warning, disabling summary')
    #utils_improc.summ_rgb('rgb_emb_embpred', rgb_emb_vis)
    
    return emb_inbound, emb_pred_inbound

def SimpleNetBlock(feat, blk_num, out_chans, istrain):
    from tensorflow.contrib.slim import conv2d, conv2d_transpose
    with tf.variable_scope('Block%d' % blk_num):
        feat = tf.pad(feat, [[0,0],[1,1],[1,1],[0,0]], 'SYMMETRIC')
        feat = conv2d(feat, out_chans*(2**blk_num), stride=2, scope='conv')
        print_shape(feat)
        feat = batch_norm(feat, istrain)
        
        feat = tf.pad(feat, [[0,0],[2,2],[2,2],[0,0]], 'SYMMETRIC')
        feat = conv2d(feat, out_chans*(2**blk_num), rate=2, scope='dilconv')
        print_shape(feat)
        feat = batch_norm(feat, istrain)
        if blk_num > 0:
            upfeat = conv2d_transpose(feat, out_chans, kernel_size=[4,4], stride=2,
                                      padding='SAME', scope='deconv')
            print_shape(upfeat)
            upfeat = batch_norm(upfeat, istrain)
        else:
            upfeat = feat
        return feat, upfeat

def SimpleNet(input, istrain, out_chans):
    slim = tf.contrib.slim
    nblocks = 2
    print_shape(input)
    B, H, W, C = input.shape.as_list()
    normalizer_fn = None
    weights_initializer = tf.truncated_normal_initializer(stddev=1e-3)
    with slim.arg_scope([slim.conv2d,
                         slim.conv2d_transpose],
                        kernel_size=3,
                        padding="VALID",
                        activation_fn=tf.nn.leaky_relu,
                        normalizer_fn=normalizer_fn,
                        weights_initializer=weights_initializer):
        upfeats = list()
        feat = input
        tf.summary.histogram(feat.name, feat)
        for blk_num in range(nblocks):
            feat, upfeat = SimpleNetBlock(feat, blk_num, out_chans, istrain)
            upfeats.append(upfeat)
        upfeat = tf.concat(upfeats, axis = 3)
        upfeat = tf.pad(upfeat, [[0,0],[2,2],[2,2],[0,0]], 'SYMMETRIC')
        emb = slim.conv2d(upfeat, out_chans, kernel_size=5, activation_fn=None, scope='conv_final')
        # emb = slim.conv2d(upfeat, out_chans, kernel_size=1, activation_fn=None,
        #                   normalizer_fn=None, scope='conv_final')
        print_shape(emb)
    return emb

def metric_loss(rgb, emb, emb_pred, emb_aug, inbound):
    ## emb,emb_pred,emb_aug -- [S,H/2,W/2,C]
    ## Use lifted_struct_loss between emb,emb_pred,emb_aug treating
    ## every s in S as a separate loss.

    # losstype = hyp.emb_loss
    # assert losstype in {'lifted', 'npairs'}
    losstype = 'lifted'
    S,H,W,C = emb.shape.as_list()
    ind,ESS = getSampleInds(rgb)
    rgb_vis = list()
    loss = 0.0
    for s in range(S):
        inbound_s = tf.gather_nd(tf.squeeze(inbound[s], axis=-1), ind[s])
        ## ind that are in-bound.
        ind_s = tf.boolean_mask(ind[s], inbound_s)
        ## Num pts in-bound could be less than 2 but lifted_struct_loss
        ## requires atleast 2 pts. So pass pts from outside bound if
        ## necessary. But only use the lifted_struct_loss if num pts
        ## in-bound is at-least 2.
        num_pts_inbound = tf.shape(ind_s)[0]
        ind_s = tf.concat([ind_s, ind[s]], axis=0)
        ind_s = ind_s[:tf.reduce_max([2,num_pts_inbound])]
        tf.summary.scalar('emb_num_pts/%02d' % s, tf.shape(ind_s)[0])
        rgb_s = tf.py_func(indAssign, [rgb[s],ind_s], tf.float32)
        rgb_vis.append(rgb_s)
        emb_s = tf.gather_nd(emb[s], ind_s)
        emb_pred_s = tf.gather_nd(emb_pred[s], ind_s)
        emb_all_s = [emb_s, emb_pred_s]
        if emb_aug is not None:
            emb_aug_s = tf.gather_nd(emb_aug[s], ind_s)
            emb_all_s.append(emb_aug_s)
        if losstype is 'lifted':
            ## labels must be [0:E]*3 where E is num pts in each of
            ## emb_s,emb_pred_s,emb_aug_s.
            labels = tf.tile(tf.range(tf.shape(emb_s)[0]), [len(emb_all_s)])
            emb_all_s = tf.concat(emb_all_s, axis=0)
            loss_s = tfml.lifted_struct_loss(labels, emb_all_s)
            ## Use lifted loss only if num pts in-bound is at-least 2.
            loss_s = tf.where(tf.greater(num_pts_inbound,1), loss_s, 0.0)
            loss += loss_s/float(S) # seems the loss scales with batchsize
            tf.summary.scalar('pix_loss/lifted_%02d' % s, loss_s)
        else:
            ## Inactive branch. Uses npairs loss.
            anchor, pos = emb_s, emb_pred_s
            labels = [tf.SparseTensor(indices=[(0,i)],
                                      values=[1],
                                      dense_shape=[1,ESS])
                      for i in range(ESS)]
            if emb_aug is not None:
                anchor = tf.concat([emb_s, emb_s], axis=0)
                pos = tf.concat([emb_pred_s, emb_aug_s], axis=0)
                labels *= 2
            loss_s = tfml.npairs_loss_multilabel(labels, anchor, pos)
            loss += loss_s
            tf.summary.scalar('pix_loss/npairs_%02d' % s, loss_s)
    rgb_vis = tf.concat(rgb_vis, axis=1)
    rgb_vis = tf.expand_dims(rgb_vis, axis=0)

    print('disabling summary')
    #utils_improc.summ_rgb('tex_pts', rgb_vis)
    
    return loss

def random_color_augs(images):
    ## Images -- [S,H,W,C]
    ## Returns color augmented versions of input images.
    images = tf.map_fn(random_color_augs_single, images, dtype=(tf.float32))
    return images

def random_color_augs_single(image):
    import preprocessor
    image += 0.5
    image = preprocessor.random_distort_color(image)
    # image = preprocessor.random_pixel_value_scale(image)
    image -= 0.5
    return image

def EmbNet(rgb, emb_pred, inbound, istrain):
    # rgb is [S,H,W,3]
    # inbound is [S,H,W,1]
    # emb_pred -- [S,H/2,W/2,C] where C is length of emb vector per pixel.

    ## Compute embs for `rgb` using EmbNet(SimpleNet) and
    ## compare/loss against `emb_pred`. Use loss only within
    ## the mask `inbound`.

    total_loss = 0.0

    with tf.variable_scope('emb'):
        print('EmbNet...')

        B, H, W, C = emb_pred.shape.as_list()
        assert(C==hyp.emb_dim)

        if hyp.emb_use_aug:
            rgb_aug = random_color_augs(rgb)
            rgb_all = tf.concat([rgb, rgb_aug], axis=0)
            emb_all = SimpleNet(rgb_all, istrain, C)
            emb, emb_aug = tf.split(emb_all, 2, axis=0)
        else:
            emb = SimpleNet(rgb, istrain, C)
            emb_aug = None
        
        rgb = tf.image.resize_bilinear(rgb, [H, W])
        inbound = tf.image.resize_nearest_neighbor(inbound, [H, W])

        # print 'SETTING INBOUND TO ALL ONES'
        # inbound = tf.ones_like(inbound)
        loss = metric_loss(rgb, emb, emb_pred, emb_aug, inbound)
        emb_pca, emb_pred_pca = emb_vis(rgb, emb, emb_pred, inbound)
        total_loss = utils_misc.add_loss(total_loss, loss,
                                         hyp.emb_coeff, 'metric')

        # embnet alone, for debug:
        # emb_vis(rgb, emb, emb_aug, inbound)
        # loss = compute_loss(rgb, emb, emb_aug, tf.ones_like(inbound))
        # total_loss = utils_misc.add_loss(total_loss, loss,
        #                                  hyp.emb_coeff, 'metric')

        smooth_loss = edge_aware_smooth_loss(emb, rgb)
        smooth_loss += edge_aware_smooth_loss(emb_pred, rgb)
        if hyp.emb_use_aug:
            smooth_loss += edge_aware_smooth_loss(emb_aug, rgb)
        total_loss = utils_misc.add_loss(total_loss, smooth_loss,
                                         hyp.emb_smooth_coeff, 'smooth')

        l1_loss_im = l1_on_chans(emb-emb_pred)
        utils_improc.summ_oned('l1_loss', l1_loss_im*inbound)
        l1_loss = reduce_masked_mean(l1_loss_im, inbound)
        total_loss = utils_misc.add_loss(total_loss, l1_loss,
                                         hyp.emb_l1_coeff, 'l1')
        
        return total_loss, emb, emb_pred, inbound, emb_pca, emb_pred_pca
        # return total_loss

