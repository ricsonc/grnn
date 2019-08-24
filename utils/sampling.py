import tensorflow as tf
import constants as const
from . import tfpy

def constant_boolean_mask_single(xxx_todo_changeme, k):
    (tensor, mask) = xxx_todo_changeme
    tensor = tf.boolean_mask(tensor, mask)
    N = tf.shape(tensor)[0]
    indices = sample_indices(k, N)
    out = tf.gather(tensor, indices)
    return out

def constant_boolean_mask(tensor, mask, k):
    return tf.map_fn(
        lambda x: constant_boolean_mask_single(x, k),
        [tensor, mask],
        dtype = tf.float32,
        parallel_iterations = const.BS
    )

def sample_with_mask_reshape(tensor, mask, sample_count):
    D = tensor.shape[-1]
    tensor = tf.reshape(tensor, (const.BS, -1, D))
    mask = tf.reshape(mask, (const.BS, -1))
    return sample_with_mask(tensor, mask, sample_count)

def sample_with_mask(tensor, mask, sample_count):
    #tensor is (BS x N x D), mask is (BS x N)
    hard_mask = mask > 0.5
    hard_float_mask = tf.cast(hard_mask, dtype = tf.float32)
    k = tf.minimum(tf.cast(tf.reduce_min(tf.reduce_sum(hard_float_mask, axis = 1)), tf.int32), sample_count)
    feats = constant_boolean_mask(tensor, hard_mask, k)
    return feats
        
def sample_indices(k, N):
    randoms = tf.random_uniform(shape = tf.stack([N]))
    topk = tf.nn.top_k(randoms, k = k, sorted = False)
    return topk.indices
