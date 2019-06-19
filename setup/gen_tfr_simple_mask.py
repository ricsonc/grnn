#!/usr/bin/env python

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
from ipdb import set_trace as st
from natsort import natsorted

const.H = 128
const.W = 128

#IN_DIR = '/home/ricson/data/ShapeNetCore.v1/all_chairs'
#IN_DIR = '/home/ricson/data/res128_double_mug/res128_mix4_all/'
#OUT_DIR = '/home/ricson/data/double_tfrs'

#IN_DIR = '/home/ricson/data/res128_double_mug/res128_4objs2/'
#OUT_DIR = '/home/ricson/data/4obj'

#IN_DIR = '/home/ricson/data/res128_double_mug/res128_3objs2/'
#OUT_DIR = '/home/ricson/data/3obj'

#IN_DIR = '/home/ricson/data/res128_double_mug/res128_multi/'
#OUT_DIR = '/home/ricson/data/multi'

#IN_DIR = '/home/ricson/data/res128_double_mug/res128_arith/'
#OUT_DIR = '/home/ricson/data/arith'

IN_DIR = '/home/ricson/data/res128_house3_all/'
OUT_DIR = '/home/ricson/data/house2'


PHIS = list(range(20, 80, 20))
# THETAS = list(range(0, 360, 20)) #for the mugs
THETAS = list(range(0, 360, 45)) #suncg

NP = len(PHIS)
NT = len(THETAS)
NV = NP*NT


listcompletedir = lambda x: [join(x,y) for y in listdir(x)]
listonlydir = lambda x: list(filter(isdir, listcompletedir(x)))


#ODERING MUST BE PRESERVED
def enum_obj_paths():
    good_paths = []
    for path in listonlydir(IN_DIR):
        stuff = listdir(path)
        good_paths.append(path)
    good_paths = natsorted(good_paths)
    print('%d data found' % len(good_paths))
    return good_paths

def parse_seg(arr):
    #red and green the two mugs
    #blue is the background
    return arr[:,:,:2]

def is_valid(img):
    #check if the center-patch is not too close to the camera
    centerh = const.H//2
    centerw = const.H//2
    r = const.H//8 #so 1/4 by 1/4 patch

    dists = 1.0 / (img + 1E-9) #convert to depth
    patch = dists[centerh-r:centerh+r, centerw-r:centerw+r]
    med_dist = np.median(patch)
    return int(med_dist > 1) and (np.max(dists) / np.min(dists) > 2.0)

def np_for_obj(obj_path_):
    view_path = obj_path_
    images = []
    angles = []
    zmaps = []
    segs = []
    valids = []
    
    for phi in PHIS:
        for theta in THETAS:
            img_name = 'RGB_%d_%d.png' % (theta, phi)
            img_path = join(view_path, img_name)
            images.append(imread(img_path).astype(np.int64))

            angles.append(np.array([phi, theta]).astype(np.float32))

            z_name = 'invZ_%d_%d.npy' % (theta, phi)
            z_path = join(view_path, z_name)
            zmaps.append(np.expand_dims(np.load(z_path), axis = 3))

            valids.append(is_valid(zmaps[-1]))

    print('... %d of %d are valid' % (sum(valids), len(valids)))
            
    return [np.stack(images, axis=0),
            np.stack(angles, axis=0),
            np.stack(zmaps, axis=0),
            np.array(valids)]

def tf_for_obj(obj_np):

    obj_np[2] = 1.0 / (obj_np[2] + 1E-9) #convert to depth
    #this is pre-scaling, so clip with these bounds
    obj_np[2] = np.clip(obj_np[2], 1.5, 2.5)

    assert obj_np[0].shape == (NV, const.H, const.W, 4)
    assert obj_np[1].shape == (NV, 2)
    assert obj_np[2].shape == (NV, const.H, const.W, 1)
    assert obj_np[3].shape == (NV,)
    
    for obj in obj_np:
        if isinstance(obj, np.ndarray):
            obj[np.isnan(obj)] = 0.0
            
    # convert everything to f32 except categories
    images, angles, zmaps, _ = list(map(np.float32, obj_np))
    valids = obj_np[-1].astype(np.uint8)
    
    example = tf.train.Example(features=tf.train.Features(feature={
        'images': _bytes_feature(images.tostring()),
        'angles': _bytes_feature(angles.tostring()),
        'zmaps': _bytes_feature(zmaps.tostring()),
        'valids': _bytes_feature(valids.tostring()),
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


def main():
    for x in enumerate(enum_obj_paths()):
        if False:
            try:
                job(x)
            except:
                print('fnf')
        else:
            job(x)
                

if __name__ == '__main__':
    main()


