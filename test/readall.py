#!/usr/bin/env python3

import sys
sys.path.append('..')
from gqn_inputs import DataReader
import tensorflow as tf
import numpy as np
import ipdb

data_reader = DataReader(
    dataset='shepard_metzler_5_parts',
    context_size=10,
    root='../gqn-dataset',
    mode='test'
)

data = data_reader.read(batch_size = 10)

with tf.train.SingularMonitoredSession() as sess:
    task = sess.run(data)

pos = task.query.context.cameras[0,:,:3]
dist = np.linalg.norm(pos, axis = 1)

print(dist)

# let's do some geometry here
pos /= 10.0/3.0

yaws = task.query.context.cameras[0,:,3]
pitches = task.query.context.cameras[0,:,4]

# ys = np.sin(pitches)
# xs = np.cos(pitches) * np.cos(yaws)
# zs = np.cos(pitches) * np.sin(yaws)
# pred_pos = np.stack([-xs, zs, -ys], axis = 1)


xs = np.cos(-pitches) * np.cos(np.pi - yaws)
ys = np.cos(-pitches) * np.sin(np.pi - yaws)
zs = np.sin(-pitches)
pred_pos = np.stack([xs, ys, zs], axis = 1)

#in other words
# x = -cos(pitch) * cos(yaw)
# y = cos(pitch) * sin(yaw)
# z = -sin(pitch)
#...wtf

#the -sin z can be explained by a vertical z and using camera pitch
#the x can be explained by a 180 deg offset??

# x = c(-p) * c(180-y)
# y = c(-p) * s(180-y)
# z = s(-p)

print(np.abs(pred_pos - pos))

