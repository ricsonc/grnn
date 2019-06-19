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
from z import read_zmap, read_norm
from bv import read_bv


#IN_DIR = '/home/ricson/data/ShapeNetCore.v1/all_chairs'
IN_DIR = '/home/ricson/data/ShapeNetCore.v1/'
OUT_DIR = '/home/ricson/data/shapenet_tfrs'


PHIS = list(range(0, 30, 10))
THETAS = list(range(0, 360, 20))


listcompletedir = lambda x: [join(x,y) for y in listdir(x)]
listonlydir = lambda x: list(filter(isdir, listcompletedir(x)))

CATEGORIES = sorted(listonlydir(IN_DIR))


def enum_obj_paths():
    good_paths = []
    for cat in CATEGORIES:
        paths = listonlydir(cat)
        for path in paths:
            stuff = listdir(path)
            if 'model.binvox' in stuff and 'model_views' in stuff:
                good_paths.append(path)
    print('%d data found' % len(good_paths))
    return good_paths[6150:]


def np_for_obj(obj_path_):
    view_path = join(obj_path_, 'model_views')
    images = []
    angles = []
    zmaps = []
    norms = []
    for phi in PHIS:
        for theta in THETAS:
            coord_name = '%d_%d_coord.exr' % (theta, phi)
            coord_path = join(view_path, coord_name)
            zmaps.append(read_zmap(coord_path, theta, phi))

            img_name = '%d_%d.png' % (theta, phi)
            img_path = join(view_path, img_name)
            images.append(imread(img_path).astype(np.int64))

            angles.append(np.array([phi, theta]).astype(np.float32))

            norm_name = '%d_%d_norm.exr' % (theta, phi)
            norm_path = join(view_path, norm_name)
            norms.append(read_norm(norm_path, theta, phi))

    binvox_path = join(obj_path_, 'model.binvox')
    binvox = read_bv(binvox_path)

    category = obj_path_[:obj_path_.rfind('/')]
    category_idx = CATEGORIES.index(category)
    
    return (np.stack(images, axis=0),
            np.stack(angles, axis=0),
            np.stack(zmaps, axis=0),
            np.stack(norms, axis=0),
            binvox, np.int32(category_idx))


def tf_for_obj(obj_np):
    assert obj_np[0].shape == (3 * 18, const.H, const.W, 4)
    assert obj_np[1].shape == (3 * 18, 2)
    assert obj_np[2].shape == (3 * 18, const.H, const.W, 1)
    assert obj_np[3].shape == (3 * 18, const.H, const.W, 3)
    assert obj_np[4].shape == (const.S, const.S, const.S)
    
    #oof owie
    #for obj in obj_np:
    #    assert not np.isnan(obj).any()

    for obj in obj_np:
        if isinstance(obj, np.ndarray):
            obj[np.isnan(obj)] = 0.0
            
    # convert everything to f32 except categories
    images, angles, zmaps, norms, vox = list(map(np.float32, obj_np[:-1]))
    category = obj_np[-1]
    
    example = tf.train.Example(features=tf.train.Features(feature={
        'images': _bytes_feature(images.tostring()),
        'angles': _bytes_feature(angles.tostring()),
        'zmaps': _bytes_feature(zmaps.tostring()),
        'norms': _bytes_feature(norms.tostring()),
        'voxel': _bytes_feature(vox.tostring()),
        'category': _bytes_feature(category.tostring()),
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
