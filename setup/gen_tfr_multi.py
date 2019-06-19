#!/usr/bin/env python2

import sys
sys.path.append('..')
import constants as const
from scipy.misc import imread
import numpy as np
import tensorflow as tf
from os import listdir
from os.path import join, isdir
from multiprocessing import Pool
from utils.tfutil import _bytes_feature
#from z import read_zmap, read_norm
from bv import read_bv

const.H = 128
const.W = 128

#IN_DIR = '/home/ricson/data/ShapeNetCore.v1/all_chairs'
IN_DIR = '/home/ricson/data/res128_double_mug/res128_mix4_all/'
OUT_DIR = '/home/ricson/data/double_tfrs'

PHIS = list(range(20, 80, 20))
THETAS = list(range(0, 360, 20))

listcompletedir = lambda x: [join(x,y) for y in listdir(x)]
listonlydir = lambda x: list(filter(isdir, listcompletedir(x)))

def enum_obj_paths():
    good_paths = []
    for path in listonlydir(IN_DIR):
        stuff = listdir(path)
        good_paths.append(path)
    print('%d data found' % len(good_paths))
    return good_paths

def parse_seg(arr):
    #red and green the two mugs
    #blue is the background
    return arr[:,:,:2]

def np_for_obj(obj_path_):
    view_path = obj_path_
    images = []
    angles = []
    zmaps = []
    segs = []
    
    for phi in PHIS:
        for theta in THETAS:
            img_name = 'RGB_%d_%d.png' % (theta, phi)
            img_path = join(view_path, img_name)
            images.append(imread(img_path).astype(np.int64))

            angles.append(np.array([phi, theta]).astype(np.float32))

            z_name = 'invZ_%d_%d.npy' % (theta, phi)
            z_path = join(view_path, z_name)
            zmaps.append(np.expand_dims(np.load(z_path), axis = 3))

            seg_name = 'amodal_%d_%d.png' % (theta, phi)
            seg_path = join(view_path, seg_name)
            segs.append(parse_seg(imread(seg_path)))

                        
    binvox_path = join(obj_path_, 'scene.binvox')
    binvox = read_bv(binvox_path)

    obj1_path = join(obj_path_, 'obj1.binvox')
    obj2_path = join(obj_path_, 'obj2.binvox')
    obj1 = read_bv(obj1_path)
    obj2 = read_bv(obj2_path)
    
    
    return [np.stack(images, axis=0),
            np.stack(angles, axis=0),
            np.stack(zmaps, axis=0),
            np.stack(segs, axis=0),
            binvox, obj1, obj2]


def tf_for_obj(obj_np):

    obj_np[2] = 1.0 / (obj_np[2] + 1E-9) #convert to depth
    #this is pre-scaling, so clip with these bounds
    obj_np[2] = np.clip(obj_np[2], 1.5, 2.5)

    assert obj_np[0].shape == (3 * 18, const.H, const.W, 4)
    assert obj_np[1].shape == (3 * 18, 2)
    assert obj_np[2].shape == (3 * 18, const.H, const.W, 1)
    assert obj_np[3].shape == (3 * 18, const.H, const.W, 2) #always have two objects...
    assert obj_np[4].shape == (const.S, const.S, const.S)
    assert obj_np[5].shape == (const.S, const.S, const.S)
    assert obj_np[6].shape == (const.S, const.S, const.S)    
    
    for obj in obj_np:
        if isinstance(obj, np.ndarray):
            obj[np.isnan(obj)] = 0.0
            
    # convert everything to f32 except categories
    images, angles, zmaps, segs, vox, obj1, obj2 = list(map(np.float32, obj_np))
    
    example = tf.train.Example(features=tf.train.Features(feature={
        'images': _bytes_feature(images.tostring()),
        'angles': _bytes_feature(angles.tostring()),
        'zmaps': _bytes_feature(zmaps.tostring()),
        'segs': _bytes_feature(segs.tostring()),
        'voxel': _bytes_feature(vox.tostring()),
        'obj1': _bytes_feature(obj1.tostring()),
        'obj2': _bytes_feature(obj2.tostring()),        
    }))
    return example


def out_path_for_obj_path(obj_path):
    return join(OUT_DIR, obj_path.split('/')[-1])


def write_tf(tfexample, path):
    compress = tf.python_io.TFRecordOptions(
        compression_type=tf.python_io.TFRecordCompressionType.GZIP)
    writer = tf.python_io.TFRecordWriter(path, options=compress)
    writer.write(tfexample.SerializeToString())
    writer.close()


def job(xxx_todo_changeme):
    (i, obj_path) = xxx_todo_changeme
    print(i, obj_path)      
    out_path = out_path_for_obj_path(obj_path)
    tfexample = tf_for_obj(np_for_obj(obj_path))
    write_tf(tfexample, out_path)


def main(mt):
    if mt:
        p = Pool(8)
        jobs = sorted(list(enumerate(enum_obj_paths())))
        p.map(job, jobs, chunksize = 1)
    else:
        for x in enumerate(enum_obj_paths()):
            job(x)


if __name__ == '__main__':
    main(True)  # set false for debug
