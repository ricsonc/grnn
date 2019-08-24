#!/usr/bin/env python

import sys
sys.path.append('..')
import constants as const  # noqa
from math import pi, cos, sin  # noqa
import numpy as np  # noqa
import OpenEXR as oe  # noqa

'''
comments about the exr files:

everything is within the (-1,1)^3 cube
everything is in the global frame, is not automatically rotated

axis order is xyz = rgb
x is horizontal, y is vertical, z is depth axis

-z is farther from camera, +z is closer 
-y is down, +y is up
-x is right, +x is left

normals -- these are outwards
-z is away from camera, +z is towards camera
-y is down, +y is up
-x is right, +x is left
'''

mm = np.matmul


def from_homo(pts):
    ws = pts[:, 3]
    pts = pts[:, 0:3] / np.expand_dims(ws, 1)
    return pts


def flatten_pts(pts):
    return np.reshape(pts, (const.H * const.W, -1))


def unflatten_pts(pts):
    return np.reshape(pts, (const.H, const.W, -1))


def apply_rot_no_homo(theta, phi, pts):
    rot_mat = rotate_matrix(-theta, phi)  # looks about right
    return from_homo(mm(make_homo(pts), rot_mat.T))


def readchannel(exrfile, channel):
    raw_bytes = exrfile.channel(channel)
    n = len(raw_bytes)
    assert n == 2 * const.H * const.W
    vals = np.frombuffer(buffer(raw_bytes), dtype=np.float16)
    return np.reshape(vals, (const.H, const.W)).astype(np.float32)


def exrread(filename):
    f = oe.InputFile(filename)
    channels = list(f.header()['channels'].keys())
    assert sorted(channels) == ['B', 'G', 'R']
    R = readchannel(f, 'R')
    B = readchannel(f, 'B')
    G = readchannel(f, 'G')
    return np.stack([R, G, B], axis=2)


def rotate_matrix(theta, phi, dist=None):
    t = theta * pi / 180.0
    p = phi * pi / 180.0

    cp = cos(p)
    sp = sin(p)
    ct = cos(t)
    st = sin(t)

    rx = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, cp, -sp, 0.0],
        [0.0, sp, cp, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    ry = np.array([
        [ct, 0.0, st, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-st, 0.0, ct, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

    rval = mm(rx, ry)

    if dist is not None:
        rval[2, 3] = -dist

    return rval


def make_homo(pts, flip=False):
    homo = np.zeros((const.H * const.W, 4))
    homo[:, 0:3] = pts
    homo[:, 3] = 1.0
    if flip:
        homo[:, 2] = -homo[:, 2]
    return homo


def coord_map_to_z_map(coord_map):
    zmap = 4.0 - coord_map[:, :, 2]  # z's are flipped
    return zmap


def read_zmap(in_coord_file, theta, phi):
    in_coord = exrread(in_coord_file)
    new_coord = unflatten_pts(apply_rot_no_homo(theta, phi, flatten_pts(in_coord)))
    return np.expand_dims(coord_map_to_z_map(new_coord), axis=3)


def read_norm(in_norm_file, theta, phi):
    in_norm = exrread(in_norm_file)
    new_norm = unflatten_pts(apply_rot_no_homo(theta, phi, flatten_pts(in_norm)))
    return new_norm
