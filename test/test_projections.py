import tensorflow as tf
import pickle
import numpy as np
import os
import utils
from utils import binvox_rw
import constants as const
output_dir = "tmp/"
# load an image to predict
with open("tmp/batch.npz", "rb") as f:
    data = pickle.load(f)

batch_id = 0
# check images 
import scipy.misc

for view_id in range(4):
  data_r = data['images'][view_id][batch_id, :, :, 0:1] 
  scipy.misc.imsave(os.path.join(output_dir, "input_r_b%d_v%d.png" %(batch_id, view_id)), 
                    np.tile(data_r, [1,1,3]))
  data_mask = data['masks'][view_id][batch_id, :, :, :] 
  scipy.misc.imsave(os.path.join(output_dir, "input_mask_b%d_v%d.png" %(batch_id, view_id)), 
                    np.tile(data_mask, [1,1,3]))

data_masks = []
for view_id in range(3):
    data_mask = data['masks'][view_id][batch_id:batch_id+1, :, :, :]
    data_mask = np.expand_dims(np.tile(data_mask, [1,1,1,2]), 0)
    data_masks.append(data_mask)
data_masks = np.concatenate(data_masks, 0)

# num_views x batch x H x W x C
ph_input_image = tf.placeholder(dtype=tf.float32, shape=data_masks.shape)
thetas = [tf.placeholder(dtype=tf.float32, shape=(const.BS)) for i in range(const.NUM_VIEWS + const.NUM_PREDS)]
phis = [tf.placeholder(dtype=tf.float32, shape=(const.BS)) for i in range(const.NUM_VIEWS + const.NUM_PREDS)]

pred_input = tf.map_fn(
  lambda x: utils.nets.unproject(x, False),
  ph_input_image, parallel_iterations = 1
)

pred_inputs = tf.split(pred_input, 3, 0)
pred_inputs = [[tf.squeeze(x, 0) for x in pred_inputs]]


def bin2theta(bin):
    return tf.cast(bin, tf.float32) * const.HDELTA + const.MINH

def bin2phi(bin):
   return tf.cast(bin, tf.float32) * const.VDELTA + const.MINV

def i2theta(view_idx):
   return bin2theta(thetas[view_idx])

def i2phi(view_idx):
  return bin2phi(phis[view_idx])

def translate_given_angles(dtheta, phi1, phi2, vox):
    #first kill elevation
    rot_mat_1 = utils.voxel.get_transform_matrix_tf([0.0]*const.BS, -phi1)
    rot_mat_2 = utils.voxel.get_transform_matrix_tf(dtheta, phi2)

    #remember to postprocess after this?
    foo = utils.voxel.rotate_voxel(vox, rot_mat_1)
    foo = utils.voxel.rotate_voxel(foo, rot_mat_2)
    return foo


def translate_views_multi(vid1s, vid2s, voxs):
    """
    rotates the 5d tensor `vox` from one viewpoint to another
    vid1s: indices of the bin corresponding to the input view
    vid2s: indices of the bin corresponding to the output view
    """
    dthetas = [i2theta(vid2) - i2theta(vid1) for (vid2, vid1) in zip(vid2s, vid1s)]
    phi1s = list(map(i2phi, vid1s))
    phi2s = list(map(i2phi, vid2s))

    dthetas = tf.stack(dthetas, 0)
    phi1s = tf.stack(phi1s, 0)
    phi2s = tf.stack(phi2s, 0)
    voxs = tf.stack(voxs, 0)
    f = lambda x: translate_given_angles(*x)
    out = tf.map_fn(f, [dthetas, phi1s, phi2s, voxs], dtype = tf.float32)
    return tf.unstack(out, axis = 0)

def aggregate_inputs(inputs):
    n = 1.0/float(len(inputs[0]))
    return [sum(input)*n for input in inputs]

pred_main_input_ = [
translate_views_multi(list(range(const.NUM_VIEWS)), [0] * (const.NUM_VIEWS), x)
   for x in pred_inputs
]

pred_main_input = aggregate_inputs(pred_main_input_)

oriented_features = [
    translate_views_multi([0] * const.NUM_PREDS,list(range(const.NUM_VIEWS, const.NUM_VIEWS + const.NUM_PREDS)),
    tf.tile(
        tf.expand_dims(feature, axis = 0),
         [const.NUM_PREDS, 1, 1, 1, 1, 1] 
    )
    )
    for feature in pred_main_input
]

projected_features = [
    utils.voxel.transformer_postprocess(
       tf.concat([
          utils.voxel.project_voxel(feature)
          for feature in features
       ],
       axis=0
       )
    )
    for features in oriented_features
]


sess = tf.Session()
# unprojection test
feed_dict = dict()
feed_dict[ph_input_image] = data_masks

for view_id in range(const.NUM_VIEWS + const.NUM_PREDS):
   feed_dict[thetas[view_id]] = data["thetas"][view_id][:const.BS]
   feed_dict[phis[view_id]] = data["phis"][view_id][:const.BS]


batch = sess.run({"thetas": thetas, "voxel": pred_input, "rotate": pred_main_input_[0], "aggregate": pred_main_input[0], "oriented":oriented_features[0][0], "projected": projected_features[0] }, feed_dict=feed_dict)

THRESHOLD = 0.7
S = batch['voxel'].shape[2]

for view_id in range(3):
    binvox_obj = binvox_rw.Voxels(
        np.transpose(batch['voxel'][view_id, batch_id, :, :, :, 0], [2, 1, 0]) > THRESHOLD,
        dims = [S, S, S],
        translate = [0.0, 0.0, 0.0], 
        scale = 1.0, 
        axis_order = 'xyz' 
    )

    with open(os.path.join(output_dir, "unproj2_b%d_v%d.binvox" %(batch_id, view_id)), "wb") as f:
        binvox_obj.write(f)    
    
    binvox_obj = binvox_rw.Voxels(
        np.transpose(batch['rotate'][view_id][batch_id, :, :, :, 0], [2, 1, 0]) > THRESHOLD,
        dims = [S, S, S],
        translate = [0.0, 0.0, 0.0], 
        scale = 1.0, 
        axis_order = 'xyz' 
    )

    with open(os.path.join(output_dir, "unproj_rotate_b%d_v%d.binvox" %(batch_id, view_id)), "wb") as f:
        binvox_obj.write(f)    

S = batch['aggregate'].shape[1]
binvox_obj = binvox_rw.Voxels(
        np.transpose(batch['aggregate'][batch_id, :, :, :, 0], [2, 1, 0]) > THRESHOLD,
        dims = [S, S, S],
        translate = [0.0, 0.0, 0.0], 
        scale = 1.0, 
        axis_order = 'xyz' 
    )

with open(os.path.join(output_dir, "aggre_%d.binvox" %(batch_id)), "wb") as f:
        binvox_obj.write(f)    

binvox_obj = binvox_rw.Voxels(
        np.transpose(batch['oriented'][batch_id, :, :, :, 0], [2, 1, 0]) > THRESHOLD,
        dims = [S, S, S],
        translate = [0.0, 0.0, 0.0], 
        scale = 1.0, 
        axis_order = 'xyz' 
    )

with open(os.path.join(output_dir, "oriented_%d.binvox" %(batch_id)), "wb") as f:
        binvox_obj.write(f)   

im = np.tile(np.expand_dims(np.mean(batch['projected'][batch_id, :, :, 20:44, 0], 2), 2), [1,1,3]).astype(np.float32)
scipy.misc.imsave(os.path.join(output_dir, "projected.png"), im)
binvox_obj = binvox_rw.Voxels(
        batch['projected'][batch_id, :, :, :, 0]> THRESHOLD,
        dims = [S, S, S],
        translate = [0.0, 0.0, 0.0], 
        scale = 1.0, 
        axis_order = 'xyz' 
    )

with open(os.path.join(output_dir, "projected_%d.binvox" %(batch_id)), "wb") as f:
        binvox_obj.write(f)    


S = data['voxel'].shape[1] 
binvox_obj = binvox_rw.Voxels(
        np.flip(np.transpose(data['voxel'][batch_id, :, :, :], [2, 1, 0]), 0) > THRESHOLD,
        dims = [S, S, S],
        translate = [0.0, 0.0, 0.0], 
        scale = 1.0, 
        axis_order = 'xyz' 
    )

with open(os.path.join(output_dir, "gt_%d.binvox" %(batch_id)), "wb") as f:
        binvox_obj.write(f)    



import ipdb; ipdb.set_trace()
