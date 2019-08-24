#!/usr/bin/env python3

import sys
sys.path.append('..')
from gqn_inputs import DataReader
import tensorflow as tf
import numpy as np

data_reader = DataReader(
    dataset='shepard_metzler_5_parts',
    context_size=10,
    root='../gqn-dataset'
)

data = data_reader.read(batch_size = 10)

with tf.train.SingularMonitoredSession() as sess:
    task = sess.run(data)



textargs = []
final = []
for i in range(10):
    
    yp = task.query.context.cameras[i,:,:]
    yaw = yp[:,0]
    pitch = yp[:,1]
    yaw = np.round(yaw*180/np.pi).astype(np.int32)    
    pitch = np.round(pitch*180/np.pi).astype(np.int32)
    
    frames = task.query.context.frames[i]
    frames = np.concatenate(list(frames), axis = 1)
    final.append(frames)
    for j in range(10):
        #textargs.append((j*64+16, i*64+16, '%d, %d' % (i,j)))
        textargs.append((j*64+16, i*64+16, '%d, %d' % (yaw[j], pitch[j])))
        
final = np.concatenate(final, axis = 0)

import matplotlib.pyplot as plt
plt.imshow(final)
for args in textargs:
    plt.text(*args, color='w')
plt.show()

#import ipdb
#ipdb.set_trace()
